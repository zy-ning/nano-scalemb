import argparse
import csv
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Engram donor probe results")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cases = []
    case_to_values = {}
    for row in rows:
        variant = row["variant_name"]
        if variant == "self":
            continue
        case = row["case_name"]
        if case not in case_to_values:
            case_to_values[case] = {}
            cases.append(case)
        case_to_values[case][variant] = float(row["delta_best_target_logit_vs_self"])

    variants = ["matched", "adversarial", "unrelated"]
    colors = {
        "matched": "#4caf50",
        "adversarial": "#ff9800",
        "unrelated": "#2196f3",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(cases)))
    width = 0.22
    offsets = {
        "matched": -width,
        "adversarial": 0.0,
        "unrelated": width,
    }

    for variant in variants:
        xs = [xi + offsets[variant] for xi in x]
        ys = [case_to_values[case].get(variant, 0.0) for case in cases]
        ax.bar(xs, ys, width=width, label=variant, color=colors[variant])

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15)
    ax.set_ylabel("Δ best-target logit vs self")
    ax.set_title("Engram donor probe: target logit modulation by donor variant")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
