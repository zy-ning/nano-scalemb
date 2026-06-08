import argparse
import csv
import os

import matplotlib.pyplot as plt

CATPPUCCIN = {
    "base": "#1e1e2e",
    "surface0": "#313244",
    "surface1": "#45475a",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "blue": "#89b4fa",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
}


ABLATION_DATA = {
    "81216": {
        "real": {"base_core": 0.2723, "val_bpb": 0.7081, "chat_core": 0.3899},
        "uniform": {"base_core": 0.2611, "val_bpb": 0.7139, "chat_core": 0.3697},
        "randomize": {"base_core": 0.2521, "val_bpb": 0.7144, "chat_core": 0.3707},
        "dense": {"base_core": 0.2460, "val_bpb": 0.7152, "chat_core": 0.3740},
    },
    "2610141822": {
        "real": {"base_core": 0.2677, "val_bpb": 0.7030, "chat_core": 0.3828},
        "uniform": {"base_core": 0.2540, "val_bpb": 0.7132, "chat_core": 0.3647},
        "randomize": {"base_core": 0.2642, "val_bpb": 0.7132, "chat_core": 0.3847},
        "dense": {"base_core": 0.2583, "val_bpb": 0.7153, "chat_core": 0.3746},
    },
}


def read_donor_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        case = row["case_name"]
        variant = row["variant_name"]
        if variant == "self":
            continue
        out.setdefault(case, {})[variant] = float(
            row["delta_best_target_logit_vs_self"]
        )
    return out


def plot_ablation(axs):
    variants = ["real", "randomize", "uniform", "dense"]
    labels = ["real", "randomize", "uniform", "dense"]
    colors = {
        "real": CATPPUCCIN["green"],
        "randomize": CATPPUCCIN["yellow"],
        "uniform": CATPPUCCIN["blue"],
        "dense": CATPPUCCIN["surface1"],
    }
    metrics = [
        ("base_core", "BASE CORE"),
        ("val_bpb", "val bpb"),
        ("chat_core", "ChatCORE"),
    ]
    metric_padding = {
        "base_core": 0.008,
        "val_bpb": 0.004,
        "chat_core": 0.008,
    }
    topologies = ["81216", "2610141822"]
    topology_labels = ["8,12,16", "2,6,10,14,18,22"]

    for ax, (metric_key, title) in zip(axs, metrics):
        width = 0.18
        x = list(range(len(topologies)))
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        for variant, offset, label in zip(variants, offsets, labels):
            ys = [
                ABLATION_DATA[topology][variant][metric_key] for topology in topologies
            ]
            xs = [xi + offset for xi in x]
            ax.bar(xs, ys, width=width, label=label, color=colors[variant])
        all_values = [
            ABLATION_DATA[topology][variant][metric_key]
            for topology in topologies
            for variant in variants
        ]
        pad = metric_padding[metric_key]
        ax.set_xticks(x)
        ax.set_xticklabels(topology_labels)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25, color=CATPPUCCIN["subtext0"])
        ax.set_facecolor(CATPPUCCIN["base"])
        ax.tick_params(colors=CATPPUCCIN["text"])
        ax.title.set_color(CATPPUCCIN["text"])
        for spine in ax.spines.values():
            spine.set_color(CATPPUCCIN["surface1"])
        if metric_key == "val_bpb":
            ax.set_ylim(max(all_values) + pad, min(all_values) - pad)
        else:
            ax.set_ylim(min(all_values) - pad, max(all_values) + pad)
    axs[0].legend(
        ncols=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(1.65, 1.22),
        frameon=False,
        labelcolor=CATPPUCCIN["text"],
    )


def plot_donor(axs, donor_data):
    variants = ["matched", "adversarial", "unrelated"]
    colors = {
        "matched": CATPPUCCIN["green"],
        "adversarial": CATPPUCCIN["red"],
        "unrelated": CATPPUCCIN["blue"],
    }
    cases = ["france_capital", "gold_symbol", "largest_planet"]
    titles = ["8,12,16 donor probe", "2,6,10,14,18,22 donor probe"]
    for ax, (topology, case_to_values), title in zip(axs, donor_data.items(), titles):
        x = list(range(len(cases)))
        width = 0.22
        offsets = {"matched": -width, "adversarial": 0.0, "unrelated": width}
        all_values = []
        for variant in variants:
            xs = [xi + offsets[variant] for xi in x]
            ys = [case_to_values.get(case, {}).get(variant, 0.0) for case in cases]
            all_values.extend(ys)
            ax.bar(xs, ys, width=width, color=colors[variant], label=variant)
        ax.axhline(0.0, color=CATPPUCCIN["subtext0"], linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=15)
        ax.set_title(title)
        ax.set_ylabel("Δ best-target logit vs self")
        ax.grid(axis="y", linestyle="--", alpha=0.25, color=CATPPUCCIN["subtext0"])
        ax.set_facecolor(CATPPUCCIN["base"])
        ax.tick_params(colors=CATPPUCCIN["text"])
        ax.title.set_color(CATPPUCCIN["text"])
        ax.yaxis.label.set_color(CATPPUCCIN["text"])
        for spine in ax.spines.values():
            spine.set_color(CATPPUCCIN["surface1"])
        ymin = min(all_values)
        ymax = max(all_values)
        pad = max(0.08, (ymax - ymin) * 0.18)
        ax.set_ylim(ymin - pad, ymax + pad)
    axs[0].legend(
        ncols=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(1.05, 1.20),
        frameon=False,
        labelcolor=CATPPUCCIN["text"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Engram rethink comparison figure"
    )
    parser.add_argument("--probe81216", required=True)
    parser.add_argument("--probe2610141822", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    donor_data = {
        "81216": read_donor_csv(args.probe81216),
        "2610141822": read_donor_csv(args.probe2610141822),
    }

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10), facecolor=CATPPUCCIN["base"])
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1], hspace=0.42, wspace=0.35)
    ablation_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    donor_axes = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]

    plot_ablation(ablation_axes)
    plot_donor(donor_axes, donor_data)

    fig.suptitle(
        "Engram: ablation and donor-probe comparison",
        fontsize=15,
        color=CATPPUCCIN["text"],
        y=1,
    )
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.07, top=0.90, hspace=0.42, wspace=0.35
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
