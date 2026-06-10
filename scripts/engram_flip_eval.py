"""Probe Engram *content* sensitivity via an in-distribution token-flip.

Unlike the donor probe (which tiles a foreign document into the Engram stream and
is therefore out-of-distribution and positionally scrambled), this probe keeps the
Engram stream a real, coherent sentence of the form the model trained on
(``engram_input_ids`` == its own context) and flips only the *entity* token.

Backbone prompt is held fixed (e.g. "The capital of France is"); the Engram stream
is a re-tokenized sentence that either agrees ("...France is", self) or swaps the
fact ("...Japan is", flip). Because the Engram right-aligns its stream to the
backbone (engram.py `_align_embeddings_to_hidden`) and the entity sits inside the
order-2/3 n-gram window at the prediction position, the flip lands exactly where the
backbone is querying.

The load-bearing readout is the *directional contrast*: flipping France->Japan
should selectively raise the matching capital (Tokyo) relative to the home capital
(Paris). The backbone always reads France, so Paris dominates in absolute logits;
the signal is the delta induced purely by the Engram flip:

    signal = [logit(flip_target) - logit(home_target)]_engram=flip
           - [logit(flip_target) - logit(home_target)]_engram=self

Run on real / randomize / uniform checkpoints: only a content-sensitive read should
produce a positive signal on `real` and ~0 on the controls.
"""

import argparse
import csv
import json
import os
from contextlib import nullcontext

import torch

from nano_scalemb.checkpoint_manager import load_model
from nano_scalemb.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from nano_scalemb.report import get_report
from scripts.engram_donor_eval import compute_token_ranks, decode_top_tokens


def first_token_id(tokenizer, text):
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError(f"Target text produced no tokens: {text!r}")
    return ids[0]


def validate_flip_alignment(prompt_ids, engram_ids, max_ngram_size, case, vname):
    """Guard the load-bearing tokenizer assumptions for a flip variant.

    A clean flip must (1) preserve length (1:1 right-alignment to the backbone),
    (2) differ from the prompt in exactly one token position (a single-token entity
    swap, not a multi-token re-segmentation), and (3) have that position fall inside
    the order-`max_ngram_size` n-gram at the prediction site, and not BE the
    predicted position itself. Multi-token entities (e.g. " Tehran" -> 3 tokens)
    silently violate (2)/(1); this turns that into a loud error.
    """
    if len(engram_ids) != len(prompt_ids):
        raise ValueError(
            f"[{case}/{vname}] engram length {len(engram_ids)} != prompt length "
            f"{len(prompt_ids)} (multi-token entity re-segmentation?)"
        )
    diff = [i for i, (a, b) in enumerate(zip(prompt_ids, engram_ids)) if a != b]
    if vname == "self":
        if diff:
            raise ValueError(f"[{case}/self] engram must equal prompt; differs at {diff}")
        return
    L = len(prompt_ids)
    if len(diff) != 1:
        raise ValueError(
            f"[{case}/{vname}] flip must differ in exactly one position; "
            f"differs at {diff} (multi-token entity?)"
        )
    pos = diff[0]
    if pos < L - max_ngram_size or pos == L - 1:
        raise ValueError(
            f"[{case}/{vname}] flipped token at pos {pos}/{L} is outside the "
            f"order-{max_ngram_size} n-gram window at the prediction position"
        )


def run_condition(model, tokenizer, prompt_ids, engram_ids, watch_ids, top_k, ctx):
    """Forward one (backbone prompt, engram stream) pair; read watch-token metrics."""
    if len(engram_ids) != len(prompt_ids):
        raise ValueError(
            f"engram stream length {len(engram_ids)} != prompt length "
            f"{len(prompt_ids)}; flip must preserve token count for 1:1 alignment"
        )
    device = model.get_device()
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    engram_tensor = torch.tensor([engram_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with ctx:
            logits = model(prompt_tensor, engram_input_ids=engram_tensor)
    next_logits = logits[0, len(prompt_ids) - 1, :].float()
    ranks = compute_token_ranks(next_logits)
    watch = {
        name: {
            "token_id": int(tid),
            "token_text": tokenizer.decode([tid]),
            "logit": float(next_logits[tid].item()),
            "rank": int(ranks[tid].item()),
        }
        for name, tid in watch_ids.items()
    }
    return {
        "watch": watch,
        "top_tokens": decode_top_tokens(tokenizer, next_logits, top_k),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Engram content sensitivity via in-distribution token flip"
    )
    parser.add_argument(
        "--source", type=str, default="base", choices=["base", "sft", "rl"]
    )
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--cases-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"]
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"]
    )
    args = parser.parse_args()

    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    model, tokenizer, meta = load_model(
        args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    if model.config.engram is None:
        raise ValueError("Loaded checkpoint has no Engram configuration")

    bos = tokenizer.get_bos_token_id()
    with open(args.cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    rows = []  # flat rows for csv
    case_summaries = {}
    for case in cases:
        prompt = case["prompt"]
        home_target = case["home_target"]
        prompt_ids = tokenizer.encode(prompt, prepend=bos)

        variants = case["variants"]
        self_var = next((v for v in variants if v["name"] == "self"), None)
        if self_var is None:
            raise ValueError(f"Case {case['name']} has no 'self' variant")
        if self_var["engram_prompt"] != prompt:
            raise ValueError(
                f"Case {case['name']} 'self' engram_prompt must equal the prompt"
            )

        # Watch set: home target + every variant's target (single-token by design).
        watch_ids = {"home": first_token_id(tokenizer, home_target)}
        for v in variants:
            if v.get("target"):
                watch_ids[f"target::{v['name']}"] = first_token_id(
                    tokenizer, v["target"]
                )

        print0("=" * 80)
        print0(f"Case: {case['name']}  | prompt: {prompt!r}  home_target: {home_target!r}")

        # Run every condition.
        max_ngram = model.config.engram.max_ngram_size
        cond = {}
        for v in variants:
            engram_ids = tokenizer.encode(v["engram_prompt"], prepend=bos)
            validate_flip_alignment(
                prompt_ids, engram_ids, max_ngram, case["name"], v["name"]
            )
            cond[v["name"]] = run_condition(
                model, tokenizer, prompt_ids, engram_ids, watch_ids,
                args.top_k, autocast_ctx,
            )

        self_w = cond["self"]["watch"]
        self_home = self_w["home"]["logit"]

        variant_summary = {}
        for v in variants:
            name = v["name"]
            w = cond[name]["watch"]
            d_home = w["home"]["logit"] - self_home
            entry = {
                "engram_prompt": v["engram_prompt"],
                "home_logit": w["home"]["logit"],
                "home_rank": w["home"]["rank"],
                "delta_home_logit_vs_self": d_home,
            }
            if v.get("target"):
                tkey = f"target::{name}"
                t_logit = w[tkey]["logit"]
                t_logit_self = self_w[tkey]["logit"]
                d_target = t_logit - t_logit_self
                # directional contrast: did the flip raise its OWN target relative
                # to the home target, beyond what self already did?
                contrast = (t_logit - w["home"]["logit"]) - (
                    self_w[tkey]["logit"] - self_home
                )
                entry.update(
                    {
                        "target_text": v["target"],
                        "target_logit": t_logit,
                        "target_rank": w[tkey]["rank"],
                        "delta_target_logit_vs_self": d_target,
                        "directional_signal": contrast,
                    }
                )
                # Surface-controlled content signal: did the fact flip raise the
                # target MORE than a non-fact filler swap in the same slot? The
                # filler carries the same non-specific tail perturbation, so this
                # isolates entity-specific content from surface (fixes the tail-shift
                # reversals where the self-baselined DiD is fooled by the tail moving).
                if "filler" in cond and tkey in cond["filler"]["watch"]:
                    sig_vs_filler = t_logit - cond["filler"]["watch"][tkey]["logit"]
                    entry["directional_signal_vs_filler"] = sig_vs_filler
                print0(
                    f"  {name:10s} | Δhome={d_home:+.4f} | Δtarget={d_target:+.4f} | "
                    f"signal={contrast:+.4f} | vs_filler="
                    f"{entry.get('directional_signal_vs_filler', float('nan')):+.4f} | "
                    f"target_rank={w[tkey]['rank']}"
                )
            else:
                print0(f"  {name:10s} | Δhome={d_home:+.4f} (control, no target)")
            variant_summary[name] = entry

            rows.append(
                {
                    "case_name": case["name"],
                    "tier": case.get("tier", ""),
                    "variant_name": name,
                    "prompt": prompt,
                    "engram_prompt": v["engram_prompt"],
                    "home_target": home_target,
                    "variant_target": v.get("target", ""),
                    "home_logit": entry["home_logit"],
                    "home_rank": entry["home_rank"],
                    "delta_home_logit_vs_self": entry["delta_home_logit_vs_self"],
                    "target_logit": entry.get("target_logit", ""),
                    "target_rank": entry.get("target_rank", ""),
                    "delta_target_logit_vs_self": entry.get(
                        "delta_target_logit_vs_self", ""
                    ),
                    "directional_signal": entry.get("directional_signal", ""),
                    "directional_signal_vs_filler": entry.get(
                        "directional_signal_vs_filler", ""
                    ),
                }
            )

        case_summaries[case["name"]] = {
            "prompt": prompt,
            "home_target": home_target,
            "tier": case.get("tier", ""),
            "variants": variant_summary,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(case_summaries, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = [
        "case_name", "tier", "variant_name", "prompt", "engram_prompt",
        "home_target", "variant_target", "home_logit", "home_rank",
        "delta_home_logit_vs_self", "target_logit", "target_rank",
        "delta_target_logit_vs_self", "directional_signal",
        "directional_signal_vs_filler",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Aggregate the load-bearing number: mean directional signal over flip variants.
    flip_signals = [
        r["directional_signal"]
        for r in rows
        if r["variant_name"] == "flip" and r["directional_signal"] != ""
    ]
    n = len(flip_signals)
    mean_signal = sum(flip_signals) / n if n else 0.0
    n_pos = sum(1 for s in flip_signals if s > 0)
    filler_signals = [
        r["directional_signal_vs_filler"]
        for r in rows
        if r["variant_name"] == "flip" and r["directional_signal_vs_filler"] != ""
    ]
    nf = len(filler_signals)
    mean_filler = sum(filler_signals) / nf if nf else 0.0
    nf_pos = sum(1 for s in filler_signals if s > 0)
    print0("=" * 80)
    print0(
        f"FLIP directional signal: mean={mean_signal:+.4f} over n={n} "
        f"({n_pos}/{n} > 0)"
    )
    print0(
        f"FLIP vs-filler (surface-controlled): mean={mean_filler:+.4f} over n={nf} "
        f"({nf_pos}/{nf} > 0)"
    )

    summary = {
        "model_tag": args.model_tag,
        "source": args.source,
        "model_step": meta["step"],
        "num_cases": len(cases),
        "flip_mean_directional_signal": mean_signal,
        "flip_n": n,
        "flip_n_positive": n_pos,
        "flip_mean_signal_vs_filler": mean_filler,
        "flip_n_positive_vs_filler": nf_pos,
        "results_json": json_path,
        "results_csv": csv_path,
    }
    get_report().log(
        section="Engram flip evaluation",
        data=[vars(args), summary, case_summaries],
    )

    compute_cleanup()


if __name__ == "__main__":
    main()
