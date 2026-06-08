"""Tier-1 Engram/MHC weight probe (CPU, no forward pass).

This codebase initialises most of the Engram + MHC routing/gating machinery to a
known value (zeros / identity / a fixed pattern, see *.init_weights()). So the
*drift of a trained weight from its init* is a direct, training-free readout of
what the model actually learned to use -- readable purely from the checkpoint's
state_dict on CPU.

Signals computed per checkpoint:

  Engram (per engram layer, init recoverable):
    - value_proj   RMS   (init 0)  -> how much the layer writes into the residual
    - short_conv   RMS   (init 0)  -> learned temporal mixing
    - post_fuse    RMS   (init 0)  -> learned stream fusion
    - branch_scales      (init 0)  -> per-stream gain = 1 + branch_scales
    - alpha/beta sinkhorn off-diag (init identity) -> cross-stream mixing learned
    - stream_key_proj ratio-to-init-RMS (init uniform) -> key routing growth
    - key_proj ratio-to-init-RMS (legacy path; expect ~1.0 = unused in mhc mode)
    - embedding row-norm distribution + slot utilisation
        (real = trained, randomize = frozen N(0,1), uniform = all rows identical)

  Backbone MHC (per transformer block, mhc_attn / mhc_mlp / mhc_engram):
    - dynamic_alpha_fn / dynamic_beta_fn norm (init 0) -> content-dependence of routing
    - pre_branch_scale / residual_scale / h_post_scale (init 1e-2) -> dynamic gain
    - static_alpha residual sinkhorn off-diag -> learned static cross-stream mixing

  MHC head:
    - dynamic_head_fn norm (init 0), head_scale (init 1e-2),
      sigmoid(head_base) (init 0.5) -> learned base stream weights

Usage:
    python -m scripts.engram_weight_probe \
        --checkpoint /path/to/ckpt_dir[:step] [more dirs...] \
        --output-dir docs/blog/weight_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# checkpoint discovery / loading
# --------------------------------------------------------------------------- #

def find_last_step(ckpt_dir: str) -> int:
    steps = []
    for fn in os.listdir(ckpt_dir):
        m = re.fullmatch(r"model_(\d+)\.pt", fn)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"no model_*.pt in {ckpt_dir}")
    return max(steps)


def load_state_dict(ckpt_dir: str, step: Optional[int]):
    if step is None:
        step = find_last_step(ckpt_dir)
    path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
    sd = torch.load(path, map_location="cpu", mmap=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    meta = {}
    meta_path = os.path.join(ckpt_dir, f"meta_{step:06d}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return sd, step, meta


# --------------------------------------------------------------------------- #
# reductions (chunked + float32 for bf16 stored tensors)
# --------------------------------------------------------------------------- #

def t_rms(w: torch.Tensor) -> float:
    return float(w.float().pow(2).mean().sqrt().item())


def t_norm(w: torch.Tensor) -> float:
    return float(w.float().norm().item())


def engram_sinkhorn(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Matches nano_scalemb.engram._sinkhorn (exp-based, row/col normalise)."""
    w = torch.exp(logits.float() - logits.float().amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w = w / w.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return w


def offdiag_fraction(doubly_stochastic: torch.Tensor) -> float:
    """1 - mean(diagonal): 0 = pure identity (no mixing), ->1 = full mixing."""
    n = doubly_stochastic.size(-1)
    diag = torch.diagonal(doubly_stochastic, dim1=-2, dim2=-1).mean().item()
    return float(1.0 - diag)


def row_norms_chunked(emb: torch.Tensor, chunk: int = 1_000_000) -> torch.Tensor:
    out = torch.empty(emb.size(0), dtype=torch.float32)
    for i in range(0, emb.size(0), chunk):
        block = emb[i : i + chunk].float()
        out[i : i + chunk] = block.pow(2).sum(dim=1).sqrt()
    return out


# common log10(row-norm) histogram grid so variants overlay
LOGHIST_LO, LOGHIST_HI, LOGHIST_BINS = -0.6, 3.3, 78


def embedding_stats(emb: torch.Tensor) -> Dict:
    """Row-norm distribution + trained-ness vs the analytic init.

    init is N(0,1) per element, so an *untouched* row has norm ~ chi(dim)
    (mean ~ sqrt(dim)). Real (trained) memory grows far past that; randomize is
    frozen at init; uniform has every row identical.
    """
    rows, dim = emb.shape
    init_mean = math.sqrt(2.0) * math.gamma((dim + 1) / 2) / math.gamma(dim / 2)
    init_std = math.sqrt(max(dim - init_mean**2, 1e-9))

    rn = row_norms_chunked(emb)
    is_uniform = bool(rn.std().item() < 1e-6)

    q = torch.quantile(
        rn[:: max(1, rows // 2_000_000)].double(),  # subsample for quantile speed
        torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64),
    )
    lo = init_mean - 3 * init_std
    hi = init_mean + 3 * init_std
    frac_outside_3s = float(((rn < lo) | (rn > hi)).float().mean().item())

    # log10 histogram on a fixed grid (handles 8.9 ... ~1000 + uniform spike)
    log_rn = rn.clamp_min(1e-6).log10().clamp(LOGHIST_LO, LOGHIST_HI)
    hist = torch.histc(log_rn, bins=LOGHIST_BINS, min=LOGHIST_LO, max=LOGHIST_HI)
    return {
        "rows": int(rows),
        "dim": int(dim),
        "init_mean": init_mean,
        "init_std": init_std,
        "rownorm_mean": float(rn.mean().item()),
        "rownorm_std": float(rn.std().item()),
        "rownorm_p1": float(q[0].item()),
        "rownorm_p50": float(q[1].item()),
        "rownorm_p99": float(q[2].item()),
        "growth_vs_init": float(rn.mean().item() / init_mean),
        "is_uniform": is_uniform,
        # fraction whose norm left the init 3-sigma band -> "trained-ness"
        "trained_frac": 0.0 if is_uniform else frac_outside_3s,
        "loghist_counts": hist.tolist(),
        "loghist_lo": LOGHIST_LO,
        "loghist_hi": LOGHIST_HI,
        "loghist_bins": LOGHIST_BINS,
    }


# --------------------------------------------------------------------------- #
# per-checkpoint analysis
# --------------------------------------------------------------------------- #

def engram_layer_ids(sd) -> List[int]:
    ids = set()
    for k in sd:
        m = re.match(r"engram_modules\.(\d+)\.", k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def backbone_layer_ids(sd) -> List[int]:
    ids = set()
    for k in sd:
        m = re.match(r"transformer\.h\.(\d+)\.mhc_attn\.", k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


D_MODEL_INIT_RMS = None  # filled per checkpoint from value_proj shape


def analyze_engram_layer(sd, lid: int, sinkhorn_iters: int) -> Dict:
    p = f"engram_modules.{lid}."
    out: Dict = {"layer": lid}

    if (w := sd.get(p + "value_proj.weight")) is not None:
        d_model = w.size(0)
        init_rms = d_model ** -0.5  # uniform(-sqrt(3)/sqrt(d), ...) RMS = d^-0.5
        out["value_proj_rms"] = t_rms(w)
    else:
        init_rms = None

    for name, key in [
        ("short_conv_rms", "short_conv.conv.weight"),
        ("post_fuse_rms", "post_fuse.weight"),
    ]:
        if (w := sd.get(p + key)) is not None:
            out[name] = t_rms(w)

    for name, key in [
        ("stream_key_proj_ratio", "stream_key_proj.weight"),
        ("key_proj_ratio", "key_proj.weight"),
    ]:
        if (w := sd.get(p + key)) is not None and init_rms:
            out[name] = t_rms(w) / init_rms

    # --- local MHC-fusion params: UNUSED when true backbone MHC is on, because
    #     the Engram then routes through forward_mhc_branches() / mhc_engram.
    #     Detect that by their drift from init (post_fuse=0, alpha/beta=identity).
    local_drift = 0.0
    if (pf := sd.get(p + "post_fuse.weight")) is not None:
        local_drift += t_norm(pf)
    if (al := sd.get(p + "alpha_logits")) is not None:
        n = al.size(-1)
        local_drift += float((al.float() - torch.eye(n)).norm().item())
    if (bs := sd.get(p + "branch_scales")) is not None:
        local_drift += float(bs.float().abs().sum().item())
    out["local_mhc_path_active"] = bool(local_drift > 1e-6)
    out["local_mhc_drift"] = local_drift

    emb = sd.get(p + "multi_head_embedding.embedding.weight")
    if emb is not None:
        out["embedding"] = embedding_stats(emb)

    return out


def analyze_backbone_layer(sd, lid: int, sinkhorn_iters: int) -> Dict:
    out: Dict = {"layer": lid}
    for slot in ("mhc_attn", "mhc_mlp", "mhc_engram"):
        p = f"transformer.h.{lid}.{slot}."
        if (da := sd.get(p + "dynamic_alpha_fn")) is None:
            continue
        rec = {
            "dyn_alpha_norm": t_norm(da),
        }
        if (db := sd.get(p + "dynamic_beta_fn")) is not None:
            rec["dyn_beta_norm"] = t_norm(db)
        for sc in ("pre_branch_scale", "residual_scale", "h_post_scale"):
            if (v := sd.get(p + sc)) is not None:
                rec[sc] = float(v.float().mean().item())
        if (sa := sd.get(p + "static_alpha")) is not None:
            # column 0 = pre-branch inject; columns 1: = residual mixing
            rec["static_alpha_offdiag"] = offdiag_fraction(
                sinkhorn_knopps(sa[..., 1:].float(), sinkhorn_iters)
            )
        out[slot] = rec
    return out


def sinkhorn_knopps(log_alpha: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Matches nano_scalemb.mhc.sinkhorn_knopps (log-domain)."""
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return log_alpha.exp()


def analyze_mhc_head(sd, sinkhorn_iters: int) -> Optional[Dict]:
    if (dh := sd.get("mhc_head.dynamic_head_fn")) is None:
        return None
    out = {"dyn_head_norm": t_norm(dh)}
    if (hs := sd.get("mhc_head.head_scale")) is not None:
        out["head_scale"] = float(hs.float().item())
    if (hb := sd.get("mhc_head.head_base")) is not None:
        out["head_base_sigmoid"] = torch.sigmoid(hb.float()).tolist()
    return out


def analyze_checkpoint(ckpt_dir: str, step: Optional[int]) -> Dict:
    sd, step, meta = load_state_dict(ckpt_dir, step)
    eng_cfg = (meta.get("model_config", {}) or {}).get("engram") or {}
    sk_iters = int(eng_cfg.get("mhc_sinkhorn_iters", 20)) if isinstance(eng_cfg, dict) else 20

    result = {
        "checkpoint_dir": ckpt_dir,
        "model_tag": os.path.basename(ckpt_dir.rstrip("/")),
        "step": step,
        "engram_layers": [],
        "backbone_layers": [],
        "mhc_head": analyze_mhc_head(sd, sk_iters),
    }
    for lid in engram_layer_ids(sd):
        result["engram_layers"].append(analyze_engram_layer(sd, lid, sk_iters))
    for lid in backbone_layer_ids(sd):
        result["backbone_layers"].append(analyze_backbone_layer(sd, lid, sk_iters))
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_summary(res: Dict) -> None:
    print("=" * 96)
    print(f"{res['model_tag']}  (step {res['step']})")
    print("-" * 96)
    print("Engram layers (live branch path: value_proj/short_conv/stream_key; "
          "init value/short_conv=0):")
    hdr = ("layer", "val_rms", "sconv", "skey/init", "loc_path",
           "emb_rn_mean", "emb_grow", "emb_train")
    print("  " + "".join(f"{h:>12}" for h in hdr))
    for L in res["engram_layers"]:
        emb = L.get("embedding", {})
        row = [
            L["layer"],
            f"{L.get('value_proj_rms', float('nan')):.4f}",
            f"{L.get('short_conv_rms', float('nan')):.4f}",
            f"{L.get('stream_key_proj_ratio', float('nan')):.2f}",
            ("on" if L.get("local_mhc_path_active") else "off"),
            f"{emb.get('rownorm_mean', float('nan')):.2f}",
            f"{emb.get('growth_vs_init', float('nan')):.1f}x",
            ("uniform" if emb.get("is_uniform") else f"{emb.get('trained_frac', float('nan')):.3f}"),
        ]
        print("  " + "".join(f"{str(c):>12}" for c in row))

    if res["mhc_head"]:
        h = res["mhc_head"]
        print(f"\nMHC head: dyn_head_norm={h['dyn_head_norm']:.3f} "
              f"head_scale={h.get('head_scale', float('nan')):.4f} "
              f"base_sigmoid={[round(x,3) for x in h.get('head_base_sigmoid', [])]}")

    bb = res["backbone_layers"]
    if bb:
        da = [b["mhc_attn"]["dyn_alpha_norm"] for b in bb if "mhc_attn" in b]
        print(f"\nBackbone MHC (mhc_attn) dyn_alpha_norm over {len(da)} layers: "
              f"min={min(da):.3f} mean={sum(da)/len(da):.3f} max={max(da):.3f}")


def write_outputs(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weight_probe.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # flat per-engram-layer CSV
    csv_path = os.path.join(out_dir, "weight_probe_engram.csv")
    fields = ["model_tag", "step", "layer", "value_proj_rms", "short_conv_rms",
              "stream_key_proj_ratio", "key_proj_ratio", "local_mhc_path_active",
              "emb_is_uniform", "emb_trained_frac", "emb_rownorm_mean",
              "emb_rownorm_std", "emb_growth_vs_init"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for res in results:
            for L in res["engram_layers"]:
                emb = L.get("embedding", {})
                w.writerow({
                    "model_tag": res["model_tag"], "step": res["step"], "layer": L["layer"],
                    "value_proj_rms": L.get("value_proj_rms"),
                    "short_conv_rms": L.get("short_conv_rms"),
                    "stream_key_proj_ratio": L.get("stream_key_proj_ratio"),
                    "key_proj_ratio": L.get("key_proj_ratio"),
                    "local_mhc_path_active": L.get("local_mhc_path_active"),
                    "emb_is_uniform": emb.get("is_uniform"),
                    "emb_trained_frac": emb.get("trained_frac"),
                    "emb_rownorm_mean": emb.get("rownorm_mean"),
                    "emb_rownorm_std": emb.get("rownorm_std"),
                    "emb_growth_vs_init": emb.get("growth_vs_init"),
                })
    print(f"\nwrote {os.path.join(out_dir, 'weight_probe.json')}")
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Tier-1 Engram/MHC weight probe (CPU)")
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="checkpoint dir, optionally dir:step. Repeatable.")
    ap.add_argument("--output-dir", default="docs/blog/weight_probe")
    args = ap.parse_args()

    results = []
    for spec in args.checkpoint:
        if ":" in spec and not spec.endswith(":"):
            ckpt_dir, step_s = spec.rsplit(":", 1)
            step = int(step_s)
        else:
            ckpt_dir, step = spec, None
        print(f"\n>>> analyzing {ckpt_dir} (step={step or 'last'}) ...", flush=True)
        res = analyze_checkpoint(ckpt_dir, step)
        print_summary(res)
        results.append(res)

    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
