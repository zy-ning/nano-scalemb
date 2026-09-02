"""Precompute a frozen SEMANTIC prior for the Engram row merge.

The over-provision->value-merge schedule plateaus because the learned values it
clusters on are only half-trained (docs/conditional_capacity_study.md; the 2x
sweet spot). This builds an ALTERNATIVE clustering signal that does not depend on
mid-training value quality: a STATIC token-embedding centroid per hash row, taken
from a fully-trained `wte`. Rows whose colliding n-grams are semantically similar
(their token-embeddings point the same way) get merged; the table then trains
already-collapsed onto that semantic partition (base_train fires the merge at
step 0 for merge_source=semantic).

It is still TOKEN-indexed: a row is a hash(token-window) bucket, exactly as in the
baseline. This is NOT the closed hidden-state idea-2 (lsh/pq addressing).

Pipeline:
  1. Load an existing n-gram count dump (scripts.build_engram_backoff_keyset
     --dump-counts) and take the top-K frequent order-3 windows. Order-2 windows
     marginalize for free: id2 = id3 % base**2.
  2. Load transformer.wte from a trained d20 checkpoint, fold raw->compressed via
     the hasher's CompressedTokenizerProjection.lookup_table, JL-project 1280->
     proj_dim (cosine-preserving; keeps the [total_rows, d] accumulator small).
  3. For each sampled window, hash it with the REAL NgramHasher (so the row it
     lands on matches the training run bit-for-bit), and scatter-add its
     token-embedding centroid (count-weighted) into feature[row].
  4. Save {feature[total_rows, proj_dim] fp16, hit_mask bool[total_rows], meta}.
     Rows with hit==0 (no sampled n-gram reached them) are flagged identity-keep
     -- base_train's merge_rows leaves them un-merged so they are not all
     collapsed into a single zero-vector survivor.

Usage:
  python -m scripts.build_engram_semantic_prior \
    --counts $NANOCHAT_BASE_DIR/engram_backoff/counts_2b_o3.npz \
    --wte-ckpt $NANOCHAT_BASE_DIR/base_checkpoints/<d20 run>/model_003814.pt \
    --slot-multiplier 30 --engram-seed 0 --proj-dim 128 --top-k 4000000 \
    --out $NANOCHAT_BASE_DIR/engram_semantic/prior_slot30_s0.pt
"""
import argparse, os, numpy as np, torch

from nano_scalemb.tokenizer import get_tokenizer
from nano_scalemb.engram import (
    EngramConfig, NgramHasher, CompressedTokenizerProjection,
    build_memory_table, ngram_orders,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True,
                    help=".npz from build_engram_backoff_keyset --dump-counts "
                    "(needs uniq_3/cnts_3; order-2 is derived by % base**2)")
    ap.add_argument("--wte-ckpt", required=True,
                    help="trained checkpoint with transformer.wte.weight [V, d]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=4_000_000,
                    help="number of most-frequent order-3 windows to scatter")
    ap.add_argument("--proj-dim", type=int, default=128,
                    help="JL target dim for the centroid feature (0 = no projection)")
    # --- must match the TARGET training run's engram geometry ---
    ap.add_argument("--slot-multiplier", type=int, default=30)
    ap.add_argument("--engram-seed", type=int, default=0)
    ap.add_argument("--engram-layers", type=str, default="2,6")
    ap.add_argument("--engram-max-ngram-size", type=int, default=3)
    ap.add_argument("--engram-heads-per-ngram", type=int, default=8)
    ap.add_argument("--engram-memory-dim", type=int, default=1280)
    ap.add_argument("--engram-mhc-num-streams", type=int, default=4)
    ap.add_argument("--no-tokenizer-compression", action="store_true")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=1_000_000,
                    help="windows hashed per batch (memory knob)")
    return ap.parse_args()


def build_hasher_and_table(args, tok, comp):
    """Reproduce the training run's hasher + shared table geometry exactly.

    Mirrors GPT._build_shared_engram_memory: the shared table's per-head vocab
    sizes are the canonical layer's primes, one contiguous run per n-gram order.
    """
    layer_ids = tuple(int(s) for s in args.engram_layers.split(",") if s.strip())
    cfg = EngramConfig(
        layer_ids=layer_ids,
        max_ngram_size=args.engram_max_ngram_size,
        n_head_per_ngram=args.engram_heads_per_ngram,
        memory_dim=args.engram_memory_dim,
        slot_multiplier=args.slot_multiplier,
        use_tokenizer_compression=not args.no_tokenizer_compression,
        seed=args.engram_seed,
        mhc_num_streams=args.engram_mhc_num_streams,
        share_memory=True,
        address_source="tokens",
    )
    vocab = tok.get_vocab_size()
    hasher = NgramHasher(cfg, vocab, comp)
    num_heads = hasher.num_hash_heads
    head_dim = cfg.memory_dim // num_heads
    canonical = hasher.resolve_layer_id(min(cfg.layer_ids))
    head_vocab_sizes = [
        hasher.prime_table[canonical][ngram_idx][head_idx]
        for ngram_idx in range(len(hasher.ngram_orders))
        for head_idx in range(cfg.n_head_per_ngram)
    ]
    table = build_memory_table(cfg, head_vocab_sizes, head_dim, args.engram_memory_dim)
    return cfg, hasher, table


def compressed_token_embeddings(args, tok, comp, device):
    """Fold raw wte [V, d] -> compressed [base, d] by mean-pooling raw rows that
    map to the same compressed id, then JL-project to proj_dim. Returns [base, p]."""
    sd = torch.load(args.wte_ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    wte = sd["transformer.wte.weight"].float()               # [V, d]
    V, d = wte.shape
    base = comp.compressed_vocab_size if comp is not None else V
    lut = comp.lookup_table if comp is not None else torch.arange(V)  # [V] raw->comp
    # mean-pool: sum embeddings per compressed id, divide by member count
    summed = torch.zeros(base, d)
    summed.index_add_(0, lut, wte)
    cnt = torch.zeros(base).index_add_(0, lut, torch.ones(V)).clamp_(min=1.0)
    E = summed / cnt.unsqueeze(1)                             # [base, d]
    if args.proj_dim and args.proj_dim < d:
        gen = torch.Generator().manual_seed(0)               # fixed JL projection
        R = torch.randn(d, args.proj_dim, generator=gen) / (args.proj_dim ** 0.5)
        E = E @ R                                             # [base, proj_dim]
    return E.to(device)


def window_tokens(wids, order, base):
    """Decode packed window-ids -> per-token compressed ids [K, order].

    Matches build_engram_backoff_keyset: id = Σ_k tok_{t-k} * base**k, so digit 0
    (units) is the CURRENT token and digit k is k steps back. Column `order-1` is
    the current token; column 0 is the oldest."""
    toks = torch.empty(wids.numel(), order, dtype=torch.long)
    w = wids.clone()
    for k in range(order):
        toks[:, order - 1 - k] = w % base
        w = w // base
    return toks


def scatter_order(order, wids, counts, E, hasher, table, base, pad_id, args, device,
                  feature, hit):
    """Hash each window of this order, add its count-weighted centroid to feature."""
    canonical = hasher.resolve_layer_id(min(hasher.cfg.layer_ids))
    K = args.chunk
    offsets = table.offsets.to(device)
    n_orders = len(hasher.ngram_orders)
    ord_idx = hasher.ngram_orders.index(order)
    hpn = hasher.cfg.n_head_per_ngram
    for i in range(0, wids.numel(), K):
        wb = wids[i : i + K]
        cb = counts[i : i + K].to(device).float()
        toks = window_tokens(wb, order, base).to(device)     # [k, order]
        # centroid of the window's token embeddings -> [k, proj_dim]
        cent = E[toks].mean(dim=1)
        # hash: build [k, order] "sequences" already in compressed space, read the
        # last position. compressed_input_ids bypasses re-compression.
        h = hasher.hash(toks, canonical, compressed_input_ids=toks)  # [k, order, n_orders*hpn]
        last = h[:, -1, :]                                    # [k, n_orders*hpn]
        # keep only this order's heads (order-major layout)
        heads = last[:, ord_idx * hpn : (ord_idx + 1) * hpn]  # [k, hpn]
        flat = heads + offsets[ord_idx * hpn : (ord_idx + 1) * hpn].unsqueeze(0)  # [k, hpn]
        # every head of this order gets the same centroid, weighted by count
        wcent = cent * cb.unsqueeze(1)                        # [k, proj_dim]
        for hh in range(hpn):
            feature.index_add_(0, flat[:, hh], wcent)
            hit.index_add_(0, flat[:, hh], cb)


def main():
    args = parse_args()
    device = torch.device(args.device)
    tok = get_tokenizer()
    comp = None if args.no_tokenizer_compression else CompressedTokenizerProjection(tok, pad_id=0)
    base = comp.compressed_vocab_size if comp is not None else tok.get_vocab_size()

    cfg, hasher, table = build_hasher_and_table(args, tok, comp)
    total_rows = table.total_rows
    proj = args.proj_dim if args.proj_dim else args.engram_memory_dim
    print(f"base {base:,} | total_rows {total_rows:,} | proj_dim {proj} | "
          f"slot {args.slot_multiplier} | seed {args.engram_seed}")

    E = compressed_token_embeddings(args, tok, comp, device)  # [base, proj]
    assert E.shape == (base, proj), (E.shape, (base, proj))

    z = np.load(args.counts)
    uniq3 = z["uniq_3"]; cnts3 = z["cnts_3"]
    k = min(args.top_k, uniq3.size)
    top = np.argpartition(cnts3, -k)[-k:]                     # top-k by count
    wid3 = torch.from_numpy(uniq3[top].astype(np.int64))
    cnt3 = torch.from_numpy(cnts3[top].astype(np.int64))
    print(f"sampled top-{k:,} order-3 windows "
          f"(of {uniq3.size:,} distinct); coverage "
          f"{cnts3[top].sum()/cnts3.sum():.1%} of order-3 occurrences")

    feature = torch.zeros(total_rows, proj, device=device)
    hit = torch.zeros(total_rows, device=device)

    # order 3 as sampled; order 2 by marginalizing id3 % base**2 (dedup + re-sum).
    scatter_order(3, wid3, cnt3, E, hasher, table, base, hasher.pad_id, args,
                  device, feature, hit)
    wid2_all = (wid3 % (base * base))
    uw2, inv = torch.unique(wid2_all, return_inverse=True)
    cnt2 = torch.zeros(uw2.numel(), dtype=torch.float64).index_add_(
        0, inv, cnt3.double()).long()
    print(f"order-2 marginal: {uw2.numel():,} distinct bigrams")
    scatter_order(2, uw2, cnt2, E, hasher, table, base, hasher.pad_id, args,
                  device, feature, hit)

    hit_mask = hit > 0
    # centroid = weighted-sum / weight (rows with hit==0 stay 0 and are identity-kept)
    feature = feature / hit.clamp(min=1.0).unsqueeze(1)
    cov = int(hit_mask.sum())
    print(f"row coverage: {cov:,}/{total_rows:,} ({cov/total_rows:.1%}) rows hit; "
          f"{total_rows-cov:,} featureless (identity-kept)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "feature": feature.half().cpu(),
        "hit_mask": hit_mask.cpu(),
        "meta": dict(total_rows=int(total_rows), proj_dim=int(proj), base=int(base),
                     slot_multiplier=int(args.slot_multiplier),
                     engram_seed=int(args.engram_seed), top_k=int(k),
                     coverage_rows=cov, wte_ckpt=args.wte_ckpt, counts=args.counts),
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
