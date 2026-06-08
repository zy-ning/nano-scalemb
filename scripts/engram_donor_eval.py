"""Probe Engram donor-stream controllability via engram_input_ids.

This script keeps the backbone prompt fixed while varying only the Engram token
stream passed through GPT.forward(..., engram_input_ids=...). It then measures
target-token logit/rank/top-k changes at the next-token prediction position.
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


def build_engram_input_ids(prompt_ids, donor_text, tokenizer, max_seq_len):
    donor_ids = tokenizer.encode(donor_text)
    if not donor_ids:
        return prompt_ids[-max_seq_len:]
    engram_ids = donor_ids[:]
    while len(engram_ids) < len(prompt_ids):
        engram_ids.extend(donor_ids)
    engram_ids = engram_ids[: len(prompt_ids)]
    if len(engram_ids) > max_seq_len:
        engram_ids = engram_ids[-max_seq_len:]
    return engram_ids


def compute_token_ranks(logits):
    sorted_ids = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(1, logits.numel() + 1, device=logits.device)
    return ranks


def summarize_target_metrics(logits, target_ids, top_k):
    ranks = compute_token_ranks(logits)
    top_ids = torch.topk(logits, k=min(top_k, logits.numel())).indices
    target_logits = logits[target_ids]
    target_ranks = ranks[target_ids]
    best_idx = int(torch.argmax(target_logits).item())
    best_token_id = int(target_ids[best_idx].item())
    best_logit = float(target_logits[best_idx].item())
    best_rank = int(target_ranks[best_idx].item())
    in_top_k = bool((top_ids.unsqueeze(1) == target_ids.unsqueeze(0)).any().item())
    return {
        "best_target_token_id": best_token_id,
        "best_target_logit": best_logit,
        "best_target_rank": best_rank,
        "mean_target_logit": float(target_logits.mean().item()),
        "min_target_rank": int(target_ranks.min().item()),
        "target_in_top_k": in_top_k,
    }


def decode_top_tokens(tokenizer, logits, top_k):
    values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
    out = []
    for score, token_id in zip(values.tolist(), indices.tolist()):
        out.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "logit": float(score),
            }
        )
    return out


def run_variant(
    model, tokenizer, prompt_ids, donor_text, target_ids, top_k, autocast_ctx
):
    engram_ids = build_engram_input_ids(
        prompt_ids, donor_text, tokenizer, model.config.sequence_len
    )
    device = model.get_device()
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    engram_tensor = torch.tensor([engram_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with autocast_ctx:
            logits = model(prompt_tensor, engram_input_ids=engram_tensor)
    next_token_logits = logits[0, len(prompt_ids) - 1, :].float()
    metrics = summarize_target_metrics(next_token_logits, target_ids, top_k)
    metrics["top_tokens"] = decode_top_tokens(tokenizer, next_token_logits, top_k)
    metrics["engram_input_token_count"] = len(engram_ids)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Engram donor controllability"
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

    with open(args.cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        prompt = case["prompt"]
        target_text = case["target_text"]
        prompt_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        target_ids = torch.tensor(
            tokenizer.encode(target_text), device=device, dtype=torch.long
        )
        if target_ids.numel() == 0:
            raise ValueError(f"Target text produced no tokens: {target_text!r}")

        print0("=" * 80)
        print0(f"Case: {case['name']}")
        print0(f"Prompt: {prompt}")
        print0(f"Target text: {target_text!r}")

        baseline_metrics = run_variant(
            model,
            tokenizer,
            prompt_ids,
            donor_text="",
            target_ids=target_ids,
            top_k=args.top_k,
            autocast_ctx=autocast_ctx,
        )
        baseline_row = {
            "case_name": case["name"],
            "variant_name": "self",
            "prompt": prompt,
            "target_text": target_text,
            **baseline_metrics,
            "delta_best_target_logit_vs_self": 0.0,
            "delta_best_target_rank_vs_self": 0,
        }
        results.append(baseline_row)
        print0(
            f"self      | best_rank={baseline_metrics['best_target_rank']:5d} | "
            f"best_logit={baseline_metrics['best_target_logit']:.4f} | "
            f"top{args.top_k}={baseline_metrics['target_in_top_k']}"
        )

        for variant in case["variants"]:
            variant_metrics = run_variant(
                model,
                tokenizer,
                prompt_ids,
                donor_text=variant["donor_text"],
                target_ids=target_ids,
                top_k=args.top_k,
                autocast_ctx=autocast_ctx,
            )
            row = {
                "case_name": case["name"],
                "variant_name": variant["name"],
                "prompt": prompt,
                "target_text": target_text,
                **variant_metrics,
                "delta_best_target_logit_vs_self": variant_metrics["best_target_logit"]
                - baseline_metrics["best_target_logit"],
                "delta_best_target_rank_vs_self": baseline_metrics["best_target_rank"]
                - variant_metrics["best_target_rank"],
            }
            results.append(row)
            print0(
                f"{variant['name']:<9} | best_rank={variant_metrics['best_target_rank']:5d} | "
                f"best_logit={variant_metrics['best_target_logit']:.4f} | "
                f"top{args.top_k}={variant_metrics['target_in_top_k']} | "
                f"Δrank={row['delta_best_target_rank_vs_self']:4d} | "
                f"Δlogit={row['delta_best_target_logit_vs_self']:.4f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = [
        "case_name",
        "variant_name",
        "prompt",
        "target_text",
        "best_target_token_id",
        "best_target_logit",
        "best_target_rank",
        "mean_target_logit",
        "min_target_rank",
        "target_in_top_k",
        "engram_input_token_count",
        "delta_best_target_logit_vs_self",
        "delta_best_target_rank_vs_self",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})

    cases_summary = {}
    for row in results:
        case_rows = cases_summary.setdefault(row["case_name"], {})
        case_rows[row["variant_name"]] = {
            "best_target_rank": row["best_target_rank"],
            "best_target_logit": row["best_target_logit"],
            "target_in_top_k": row["target_in_top_k"],
            "delta_best_target_logit_vs_self": row["delta_best_target_logit_vs_self"],
            "delta_best_target_rank_vs_self": row["delta_best_target_rank_vs_self"],
        }

    summary = {
        "model_tag": args.model_tag,
        "source": args.source,
        "top_k": args.top_k,
        "num_cases": len(cases),
        "results_json": json_path,
        "results_csv": csv_path,
    }
    get_report().log(
        section="Engram donor evaluation",
        data=[vars(args), {"model_step": meta["step"]}, summary, cases_summary],
    )

    compute_cleanup()


if __name__ == "__main__":
    main()
