"""Tier-2 Engram/MHC forward-pass probe (cheap inference signals).

The Tier-1 weight probe (scripts/engram_weight_probe.py) reads *what the model
learned* straight from the checkpoint -- weights only, no forward pass. This
Tier-2 probe runs a single forward pass over a small fixed batch of real
validation tokens and reads *what the model actually does at inference*: the
signals that only exist once activations flow.

It is "cheap" in the Tier-1 sense -- no training, no backprop, a handful of
batches on CPU -- but it sees things the static probe cannot:

  Engram read gate (faithful per-stream path, _mhc_gated_values):
    gate_t,s = sigmoid( <RMSNorm(h_t), RMSNorm(k_t,s)> / sqrt(d) ) in (0, 1)
    - gate_mean        -> how far the Engram opens its read on real text
    - gate_std         -> selectivity: does the gate respond to content, or is
                          it flat? (uniform table -> identical keys -> ~0 std)
    - per-stream gate_mean / gate_std -> which residual streams the read serves

  Engram contribution (forward_mhc_branches output, per stream):
    - engram_out_rms   -> the magnitude actually written into the residual
                          (the inference-time counterpart of Tier-1 value_proj
                          RMS; uniform collapses value_proj -> ~0 here)

  MHC head final collapse (MHCHead.forward, all checkpoints incl. baseline):
    - head_mix per stream -> the learned per-stream weighting at the LM head

  Summary:
    - mean_nll / bpb   -> next-token loss on the probe batch

The gate / head-mix tensors are recomputed inside light wrappers using each
module's own trained submodules (mirroring engram._mhc_gated_values and
mhc.MHCHead.forward, exactly as the weight probe mirrors the two sinkhorns).
The wrappers always return the model's true output, so the forward pass -- and
hence the reported loss -- is unchanged.

Usage:
    python -m scripts.engram_forward_probe \
        --checkpoint /path/to/ckpt_dir[:step] [more dirs...] \
        --device cpu --batch-size 8 --seq-len 512 --num-batches 1 \
        --output-dir docs/blog/forward_probe
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from nano_scalemb.checkpoint_manager import build_model, find_last_step
from nano_scalemb.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nano_scalemb.tokenizer import get_token_bytes, get_tokenizer


# --------------------------------------------------------------------------- #
# probe batch (fixed across checkpoints for an apples-to-apples comparison)
# --------------------------------------------------------------------------- #

def load_probe_batch(batch_size: int, seq_len: int, num_batches: int):
    """Grab the first `num_batches` val batches once, on CPU, deterministically.

    The bestfit loader is deterministic from a cold start (single process, no
    resume), so every checkpoint sees the exact same tokens.
    """
    tokenizer = get_tokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, seq_len, "val", device="cpu"
    )
    inputs, targets = [], []
    for _ in range(num_batches):
        x, y = next(loader)
        inputs.append(x)
        targets.append(y)
    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


# --------------------------------------------------------------------------- #
# capture wrappers (recompute internal tensors via each module's own weights;
# the model's true output is always returned unchanged)
# --------------------------------------------------------------------------- #

def _attach_engram_capture(engram, store: Dict):
    """Wrap forward_mhc_branches to log gate + contribution stats per call.

    Mirrors engram.Engram._mhc_gated_values for the gate; the contribution RMS
    is read from the module's real output.
    """
    orig = engram.forward_mhc_branches
    d_model = engram.d_model
    streams = engram.mhc_num_streams

    def wrapped(x, input_ids, compressed_input_ids=None):
        out = orig(x, input_ids, compressed_input_ids=compressed_input_ids)
        with torch.no_grad():
            emb = engram._compute_embeddings(
                input_ids, compressed_input_ids=compressed_input_ids
            )
            emb = engram._align_embeddings_to_hidden(x, emb)
            query = engram.stream_query_norm(x).unsqueeze(2)  # [B, T, 1, D]
            keys = engram.stream_key_proj(emb).view(
                emb.size(0), emb.size(1), streams, d_model
            )
            keys = F.rms_norm(keys, (d_model,))
            gates = torch.sigmoid(
                (query * keys).sum(dim=-1, keepdim=True) / math.sqrt(d_model)
            )  # [B, T, S, 1]
            _record_engram(store, gates.float(), out.float(), x.float())
        return out

    engram.forward_mhc_branches = wrapped
    return orig


def _record_engram(store, gates, out, x):
    # gates: [B, T, S, 1]; out: [B, T, S, D]; x: [B, T, D]
    g = gates.squeeze(-1)  # [B, T, S]
    flat = g.reshape(-1)
    q = torch.quantile(
        flat[:: max(1, flat.numel() // 200_000)],
        torch.tensor([0.10, 0.50, 0.90]),
    )
    out_rms_per_stream = out.pow(2).mean(dim=(0, 1, 3)).sqrt()  # [S]
    store.setdefault("count", 0)
    store["count"] += 1
    store.setdefault("samples", []).append(flat.clone())
    acc = store.setdefault("acc", {
        "gate_mean": 0.0, "gate_sq": 0.0,
        "gate_p10": 0.0, "gate_p50": 0.0, "gate_p90": 0.0,
        "gate_mean_s": torch.zeros(g.size(-1)),
        "gate_std_s": torch.zeros(g.size(-1)),
        "out_rms": 0.0,
        "out_rms_s": torch.zeros(g.size(-1)),
        "x_rms": 0.0,
    })
    acc["gate_mean"] += float(flat.mean())
    acc["gate_sq"] += float(flat.pow(2).mean())
    acc["gate_p10"] += float(q[0])
    acc["gate_p50"] += float(q[1])
    acc["gate_p90"] += float(q[2])
    acc["gate_mean_s"] += g.mean(dim=(0, 1))
    acc["gate_std_s"] += g.std(dim=(0, 1))
    acc["out_rms"] += float(out.pow(2).mean().sqrt())
    acc["out_rms_s"] += out_rms_per_stream
    acc["x_rms"] += float(x.pow(2).mean().sqrt())


GATE_SAMPLE_CAP = 2500  # downsample for compact violin distributions in JSON


def _finalize_engram(store) -> Optional[Dict]:
    n = store.get("count", 0)
    if n == 0:
        return None
    a = store["acc"]
    gate_mean = a["gate_mean"] / n
    gate_var = max(a["gate_sq"] / n - gate_mean ** 2, 0.0)
    gate_std_s = a["gate_std_s"] / n  # per-stream std *across tokens*

    # deterministic stride subsample of the raw gate values for violins
    samples = torch.cat(store["samples"])
    stride = max(1, samples.numel() // GATE_SAMPLE_CAP)
    gate_samples = [round(v, 4) for v in samples[::stride].tolist()]

    return {
        "gate_mean": gate_mean,
        "gate_std": math.sqrt(gate_var),
        # selectivity: how much the gate varies across tokens (mean over
        # streams of the across-token std). ~0 => gate ignores content.
        "gate_selectivity": float(gate_std_s.mean()),
        "gate_p10": a["gate_p10"] / n,
        "gate_p50": a["gate_p50"] / n,
        "gate_p90": a["gate_p90"] / n,
        "gate_mean_per_stream": (a["gate_mean_s"] / n).tolist(),
        "gate_std_per_stream": gate_std_s.tolist(),
        "engram_out_rms": a["out_rms"] / n,
        "engram_out_rms_per_stream": (a["out_rms_s"] / n).tolist(),
        "branch_input_rms": a["x_rms"] / n,
        "gate_samples": gate_samples,
    }


def _attach_head_capture(head, store: Dict):
    """Wrap MHCHead.forward to log the per-stream mix; mirrors MHCHead.forward."""
    if head is None or head.num_residual_streams == 1:
        return None
    orig = head.forward
    streams = head.num_residual_streams
    dim = head.dim

    def wrapped(residuals):
        with torch.no_grad():
            bs, seq, d = residuals.shape
            batch = bs // streams
            r = residuals.view(batch, streams, seq, d).transpose(1, 2)
            normed = r.reshape(batch, seq, streams * d)
            normed = F.rms_norm(
                normed, head.norm.normalized_shape,
                head.norm.weight.to(normed.dtype), head.norm.eps,
            )
            logits = normed @ head.dynamic_head_fn.to(normed.dtype)
            mix = torch.sigmoid(
                logits * head.head_scale.to(logits.dtype)
                + head.head_base.to(logits.dtype)
            ).float()  # [batch, seq, streams]
            store.setdefault("count", 0)
            store["count"] += 1
            acc = store.setdefault("acc", {
                "mix_mean_s": torch.zeros(streams),
                "mix_std_s": torch.zeros(streams),
            })
            acc["mix_mean_s"] += mix.mean(dim=(0, 1))
            acc["mix_std_s"] += mix.std(dim=(0, 1))
        return orig(residuals)

    head.forward = wrapped
    return orig


def _finalize_head(store) -> Optional[Dict]:
    n = store.get("count", 0)
    if n == 0:
        return None
    a = store["acc"]
    return {
        "head_mix_mean_per_stream": (a["mix_mean_s"] / n).tolist(),
        "head_mix_std_per_stream": (a["mix_std_s"] / n).tolist(),
    }


# --------------------------------------------------------------------------- #
# per-checkpoint forward pass
# --------------------------------------------------------------------------- #

def analyze_checkpoint(
    ckpt_dir: str, step: Optional[int], inputs, targets, device, autocast_ctx,
    token_bytes,
) -> Dict:
    if step is None:
        step = find_last_step(ckpt_dir)
    model, _tokenizer, meta = build_model(ckpt_dir, step, device, phase="eval")
    cfg = model.config

    engram_stores: Dict[int, Dict] = {}
    head_store: Dict = {}
    originals = []
    if cfg.engram is not None:
        for lid, module in model.engram_modules.items():
            store: Dict = {}
            engram_stores[int(lid)] = store
            originals.append(("engram", module, _attach_engram_capture(module, store)))
    head_orig = _attach_head_capture(model.mhc_head, head_store)
    if head_orig is not None:
        originals.append(("head", model.mhc_head, head_orig))

    # forward in chunks over the probe batch; accumulate loss in bytes for bpb
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0.0
    chunk = inputs.size(0)
    tb = token_bytes.to(device)
    with torch.no_grad():
        for i in range(0, inputs.size(0), chunk):
            x = inputs[i : i + chunk].to(device)
            y = targets[i : i + chunk].to(device)
            with autocast_ctx:
                logits = model(x)  # [B, T, V]
            logits = logits.float()
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                ignore_index=-1, reduction="sum",
            )
            ntok = int((y.view(-1) != -1).sum())
            total_nll += float(nll)
            total_tokens += ntok
            total_bytes += float(tb[y.view(-1).clamp_min(0)].sum())

    # restore wrapped methods
    for kind, module, orig in originals:
        if kind == "engram":
            module.forward_mhc_branches = orig
        else:
            module.forward = orig

    mean_nll = total_nll / max(total_tokens, 1)
    bpb = total_nll / (math.log(2) * max(total_bytes, 1e-9))

    result = {
        "checkpoint_dir": ckpt_dir,
        "model_tag": os.path.basename(ckpt_dir.rstrip("/")),
        "step": step,
        "ablation_mode": (cfg.engram.ablation_mode if cfg.engram else "baseline"),
        "engram_layer_ids": (list(cfg.engram.layer_ids) if cfg.engram else []),
        "num_streams": (cfg.mhc.num_streams if cfg.mhc else 1),
        "probe_tokens": total_tokens,
        "mean_nll": mean_nll,
        "bpb": bpb,
        "engram_layers": [],
        "mhc_head": _finalize_head(head_store),
    }
    for lid in sorted(engram_stores):
        rec = _finalize_engram(engram_stores[lid])
        if rec is not None:
            rec = {"layer": lid, **rec}
            result["engram_layers"].append(rec)

    del model
    gc.collect()
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_summary(res: Dict) -> None:
    print("=" * 96)
    print(f"{res['model_tag']}  (step {res['step']}, ablation={res['ablation_mode']})")
    print(f"  probe: {res['probe_tokens']} tokens   mean_nll={res['mean_nll']:.4f}   "
          f"bpb={res['bpb']:.4f}")
    if res["engram_layers"]:
        print("-" * 96)
        hdr = ("layer", "gate_mean", "gate_std", "g_p10", "g_p90",
               "out_rms", "x_rms")
        print("  " + "".join(f"{h:>11}" for h in hdr))
        for L in res["engram_layers"]:
            row = [
                L["layer"],
                f"{L['gate_mean']:.4f}", f"{L['gate_std']:.4f}",
                f"{L['gate_p10']:.4f}", f"{L['gate_p90']:.4f}",
                f"{L['engram_out_rms']:.4f}", f"{L['branch_input_rms']:.3f}",
            ]
            print("  " + "".join(f"{str(c):>11}" for c in row))
    if res["mhc_head"]:
        mix = res["mhc_head"]["head_mix_mean_per_stream"]
        print(f"  MHC head mix (mean/stream): {[round(x, 3) for x in mix]}")


def write_outputs(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "forward_probe.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "forward_probe_engram.csv")
    fields = ["model_tag", "step", "ablation_mode", "layer", "gate_mean",
              "gate_std", "gate_p10", "gate_p50", "gate_p90", "engram_out_rms",
              "branch_input_rms", "mean_nll", "bpb"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for res in results:
            for L in res["engram_layers"]:
                w.writerow({
                    "model_tag": res["model_tag"], "step": res["step"],
                    "ablation_mode": res["ablation_mode"], "layer": L["layer"],
                    "gate_mean": L["gate_mean"], "gate_std": L["gate_std"],
                    "gate_p10": L["gate_p10"], "gate_p50": L["gate_p50"],
                    "gate_p90": L["gate_p90"], "engram_out_rms": L["engram_out_rms"],
                    "branch_input_rms": L["branch_input_rms"],
                    "mean_nll": res["mean_nll"], "bpb": res["bpb"],
                })
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Tier-2 Engram/MHC forward-pass probe (cheap inference signals)"
    )
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="checkpoint dir, optionally dir:step. Repeatable.")
    ap.add_argument("--output-dir", default="docs/blog/forward_probe")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-batches", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    ptdtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    autocast_ctx = (
        torch.amp.autocast(device_type=args.device, dtype=ptdtype)
        if args.device == "cuda" else nullcontext()
    )

    print(f"loading probe batch: B={args.batch_size} T={args.seq_len} "
          f"x {args.num_batches} batch(es) ...", flush=True)
    inputs, targets = load_probe_batch(
        args.batch_size, args.seq_len, args.num_batches
    )
    token_bytes = get_token_bytes(device="cpu")
    print(f"probe batch: {tuple(inputs.shape)}  ({inputs.numel()} tokens)", flush=True)

    results = []
    for spec in args.checkpoint:
        if ":" in spec and not spec.endswith(":"):
            ckpt_dir, step_s = spec.rsplit(":", 1)
            step = int(step_s)
        else:
            ckpt_dir, step = spec, None
        print(f"\n>>> {ckpt_dir} (step={step or 'last'}) ...", flush=True)
        res = analyze_checkpoint(
            ckpt_dir, step, inputs, targets, device, autocast_ctx, token_bytes
        )
        print_summary(res)
        results.append(res)

    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
