"""Precompute a frozen Engram hash-backoff keyset.

For a chosen n-gram order, keep the "worth their own row" windows and let every
other window back off to its per-token (*,*,t) key at train time. Selection is
per-ending-token nucleus (top-p): for each ending token t, sort its windows
(a,...,t) by corpus count and keep the smallest set whose cumulative count-share
reaches p. A diffuse continuation distribution keeps more keys, a peaked one fewer.

  --mode topp     : one p for every token.
  --mode fw_topp  : p_t rises with the token's own frequency, so more frequent
                    ending tokens keep more of their windows:
                      p_t = p + (1-p) * (log1p(f_t)-log1p(fmin))
                                        / (log1p(fmax)-log1p(fmin))

Output (torch.save): {"keysets": {order: sorted int64 window-ids}, "meta": {...}}.
The window-id encoding (id = Σ_k token_{t-k} * base^k, base = compressed vocab)
MUST match nano_scalemb.engram.NgramHasher; this script and the hasher share it.

Why ~100M tokens (not the 10M diagnostic): the per-token count threshold needs a
stable estimate; a small sample would prune learnable windows as if rare.

Usage:
  python -m scripts.build_engram_backoff_keyset --mode topp --p 0.6 \
      --out $NANOCHAT_BASE_DIR/engram_backoff/ks_topp.pt
  # match a key budget instead of fixing p (binary-search p to hit --target-keys):
  python -m scripts.build_engram_backoff_keyset --mode fw_topp --target-keys 106000 \
      --out .../ks_fw.pt
"""
import argparse, os, numpy as np, torch

from nano_scalemb.tokenizer import get_tokenizer
from nano_scalemb.engram import CompressedTokenizerProjection
from nano_scalemb.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["topp", "fw_topp", "allseen"], required=True,
                    help="allseen = keep every window observed in the sample (no "
                    "selection); backs off ONLY keyset-unseen windows. Isolates the "
                    "unseen-backoff component. Best built on a LARGE sample so 'seen' "
                    "≈ 'learnable'.")
    ap.add_argument("--p", type=float, default=0.6, help="nucleus threshold (floor for fw_topp)")
    ap.add_argument("--target-keys", type=int, default=0,
                    help="if >0, binary-search --p so the total kept count ~= this")
    ap.add_argument("--orders", type=str, default="3",
                    help="comma list of n-gram orders to build keysets for")
    ap.add_argument("--n-tokens", type=int, default=100_000_000)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--device-batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--slot-multiplier", type=int, default=15,
                    help="only for reporting the resulting table load")
    ap.add_argument("--no-tokenizer-compression", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dump-counts", type=str, default="",
                    help="after scanning, save (uniq,counts,unigram) to this .npz so "
                    "other keysets can be built from the same pass with --from-counts")
    ap.add_argument("--from-counts", type=str, default="",
                    help="load counts from a --dump-counts .npz and skip the scan")
    return ap.parse_args()


def save_counts(path, counts, uni, got, orders):
    arrs = {"uni": uni, "got": np.int64(got), "orders": np.array(orders)}
    for o, (u, c) in counts.items():
        arrs[f"uniq_{o}"] = u; arrs[f"cnts_{o}"] = c
    np.savez(path, **arrs)


def load_counts(path):
    z = np.load(path)
    orders = [int(o) for o in z["orders"]]
    counts = {o: (z[f"uniq_{o}"], z[f"cnts_{o}"]) for o in orders}
    return counts, z["uni"], int(z["got"])


def collect_counts(orders, n_tokens, split, dbs, seq_len, comp, base, cpad):
    """Return {order: (uniq_ids, counts)} and the per-token unigram counts."""
    tok = get_tokenizer()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, dbs, seq_len, split=split, device="cpu", resume_state_dict=None)
    per_order_ids = {o: [] for o in orders}
    uni = np.zeros(base, dtype=np.int64)
    got = 0
    while got < n_tokens:
        x, _, _ = next(loader)
        x = x.clone().to(torch.int64)          # loader reuses one in-place buffer
        cx = comp.compress(x) if comp is not None else x
        B, T = cx.shape
        got += x.numel()
        cur = cx.reshape(-1).numpy()
        np.add.at(uni, cur, 1)
        for o in orders:
            ids = cx.clone().to(torch.int64); radix = 1
            for k in range(1, o):
                radix *= base
                sh = torch.full_like(cx, cpad); sh[:, k:] = cx[:, :T - k]
                ids = ids + sh.to(torch.int64) * radix
            per_order_ids[o].append(ids.reshape(-1).numpy())
    out = {}
    for o in orders:
        allids = np.concatenate(per_order_ids[o])
        uniq, cnts = np.unique(allids, return_counts=True)
        out[o] = (uniq, cnts)
    return out, uni, got


def sorted_stats(uniq, cnts, base, uni):
    """Precompute the per-ending-token sort once; both modes select off it.

    Returns the count-desc sort order and, aligned to it: frac_before (cumulative
    count-share strictly before each window within its ending-token group),
    rank_in_group (0-based position by desc count), and freq_t (the ending token's
    own unigram count)."""
    end_tok = uniq % base
    order = np.lexsort((-cnts, end_tok))               # by token, then desc count
    et_s, cnt_s = end_tok[order], cnts[order].astype(np.float64)
    grp_start = np.empty(et_s.size, dtype=bool)
    grp_start[0] = True; grp_start[1:] = et_s[1:] != et_s[:-1]
    grp_id = np.cumsum(grp_start) - 1
    first = np.maximum.accumulate(np.where(grp_start, np.arange(et_s.size), 0))
    grp_tot = np.zeros(grp_id[-1] + 1); np.add.at(grp_tot, grp_id, cnt_s)
    csum = np.cumsum(cnt_s)
    grp_base = csum[first] - cnt_s[first]              # cumulative before each group
    cum_before = (csum - cnt_s) - grp_base            # cumulative strictly before self
    frac_before = cum_before / grp_tot[grp_id]
    rank_in_group = np.arange(et_s.size) - first
    return dict(order=order, frac_before=frac_before, rank=rank_in_group,
                freq_t=uni[et_s].astype(np.float64), size=uniq.size)


def keep_mask(stats, mode, param):
    """Boolean mask over distinct windows for a given selection parameter.

    topp    : keep the count-nucleus reaching cumulative share `param` (= p).
    fw_topp : per-token key budget k_t = floor(param * freq_t) (freq-weighted
              top-k), so a more frequent ending token keeps more of its windows;
              `param` is a global scale solved to hit the key budget."""
    if mode == "topp":
        keep_sorted = stats["frac_before"] < param      # include the crossing element
    else:  # fw_topp
        keep_sorted = stats["rank"] < np.floor(param * stats["freq_t"])
    keep = np.zeros(stats["size"], dtype=bool)
    keep[stats["order"]] = keep_sorted
    return keep


def main():
    args = parse_args()
    orders = [int(o) for o in args.orders.split(",") if o.strip()]
    tok = get_tokenizer(); vocab = tok.get_vocab_size()
    comp = None if args.no_tokenizer_compression else CompressedTokenizerProjection(tok, pad_id=0)
    base = vocab if comp is None else comp.compressed_vocab_size
    cpad = 0 if comp is None else comp.compressed_pad_id
    for o in orders:
        assert base ** o < 2 ** 62, f"order {o} base {base} overflows int64 window-id"
    print(f"tokenizer vocab {vocab:,} | base {base:,} | orders {orders} | mode {args.mode}")

    if args.from_counts:
        counts, uni, got = load_counts(args.from_counts)
        print(f"loaded counts from {args.from_counts} ({got:,} tokens scanned)")
    else:
        counts, uni, got = collect_counts(orders, args.n_tokens, args.split,
                                          args.device_batch_size, args.seq_len, comp, base, cpad)
        print(f"scanned {got:,} tokens")
    if args.dump_counts:
        save_counts(args.dump_counts, counts, uni, got, orders)
        print(f"dumped counts -> {args.dump_counts}")

    param = args.p
    if args.mode == "allseen":
        # No selection: keep every observed window. Skip the (expensive on a huge
        # distinct set) per-token sort entirely.
        stats = None
    else:
        stats = {o: sorted_stats(u, c, base, uni) for o, (u, c) in counts.items()}

        def total_kept(param):
            return sum(int(keep_mask(stats[o], args.mode, param).sum()) for o in counts)

        # topp -> p in [0,1]; fw_topp -> a global scale s>=0 with per-token budget
        # floor(s * freq_t). Both monotone in the kept count -> bracketed bisection.
        if args.target_keys > 0:
            lo, hi = 0.0, 1.0
            if args.mode != "topp":                     # grow hi until it overshoots
                while total_kept(hi) < args.target_keys and hi < 1e6:
                    hi *= 2
            for _ in range(40):
                param = (lo + hi) / 2
                if total_kept(param) < args.target_keys:
                    lo = param                          # keep more
                else:
                    hi = param
            knob = "p" if args.mode == "topp" else "s"
            print(f"matched budget: {knob}={param:.6g} -> {total_kept(param):,} keys "
                  f"(target {args.target_keys:,})")

    keysets, meta_orders = {}, {}
    P_per_head = base * args.slot_multiplier
    for o, (uniq, cnts) in counts.items():
        keep = (np.ones(uniq.size, dtype=bool) if args.mode == "allseen"
                else keep_mask(stats[o], args.mode, param))
        kept_ids = np.sort(uniq[keep])
        keysets[o] = torch.from_numpy(kept_ids.astype(np.int64))
        cov = cnts[keep].sum() / cnts.sum()
        meta_orders[o] = dict(distinct=int(uniq.size), kept=int(kept_ids.size),
                              coverage=float(cov), approx_load=float(kept_ids.size / P_per_head))
        print(f"  order {o}: {uniq.size:,} distinct -> keep {kept_ids.size:,} "
              f"({kept_ids.size/uniq.size:.1%}) | occ coverage {cov:.1%} | "
              f"approx load {kept_ids.size/P_per_head:.1%} of a slot={args.slot_multiplier} table")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"keysets": keysets,
                "meta": dict(mode=args.mode, p=float(param), base=int(base),
                             orders=orders, n_tokens=int(got), split=args.split,
                             target_keys=int(args.target_keys), per_order=meta_orders)},
               args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
