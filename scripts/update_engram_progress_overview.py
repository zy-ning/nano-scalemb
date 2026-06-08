import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


REPO_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_DIR / "runs" / "reports"
OVERVIEW_DIR = REPORTS_DIR / "engram_progress_overview"

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

METRIC_COLUMNS = [
    "base_core",
    "val_bpb",
    "chatcore",
    "arc_easy",
    "arc_challenge",
    "mmlu",
    "gsm8k",
    "humaneval",
]

DISPLAY_NAMES = {
    "base_core": "BASE CORE",
    "val_bpb": "val bpb",
    "chatcore": "ChatCORE",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
}


def to_float(value: str):
    if value is None:
        return None
    value = value.strip().strip("`")
    if not value or value.lower() == "none":
        return None
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    try:
        return float(value)
    except ValueError:
        return None


def parse_sweep_name(path: Path):
    return path.parent.name


def parse_run_tag_to_model(run_tag: str, sweep_name: str):
    if run_tag.startswith("nano-engram"):
        return run_tag
    tag = run_tag.strip("`")
    return f"nano-engram-d24-sweep-{tag}-{sweep_name.replace('engram_sweep_', '')}"


def infer_layers_from_model(model: str):
    match = re.search(r"sweep-([0-9,]+|[0-9]+(?:[0-9]{1,2})*)-dim", model)
    if match and "," in match.group(1):
        return match.group(1)
    match = re.search(r"sweep-([0-9]+)-dim", model)
    if not match:
        return ""
    compact = match.group(1)
    if "," in compact:
        return compact

    candidates = [str(i) for i in range(1, 25)]

    def helper(remaining: str, previous: int):
        if not remaining:
            return []
        for candidate in candidates:
            if not remaining.startswith(candidate):
                continue
            value = int(candidate)
            if value <= previous:
                continue
            tail = helper(remaining[len(candidate) :], value)
            if tail is not None:
                return [candidate] + tail
        return None

    parsed = helper(compact, 0)
    if parsed is None:
        return compact
    return ",".join(parsed)


def normalize_layers(value: str):
    value = value.strip().strip("`")
    if not value:
        return value
    if "," in value:
        return ",".join(part.strip() for part in value.split(",") if part.strip())
    return value


def resolve_experiment_info(experiment_rows, run_tag: str):
    if run_tag in experiment_rows:
        return experiment_rows[run_tag]
    stripped = run_tag.strip().strip("`")
    for key, value in experiment_rows.items():
        key_stripped = key.strip().strip("`")
        key_suffix = key_stripped.split("sweep-", 1)[-1]
        if key_stripped == stripped or key_suffix == stripped:
            return value
    return {}


def parse_markdown_table(lines, start_index):
    header = [cell.strip() for cell in lines[start_index].strip().strip("|").split("|")]
    rows = []
    index = start_index + 2
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(header):
            break
        rows.append(dict(zip(header, cells)))
        index += 1
    return header, rows, index


def parse_early_summary(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sweep_name = parse_sweep_name(path)
    experiment_rows = {}
    results = []

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("| Run tag | Engram layers |"):
            _, rows, index = parse_markdown_table(lines, index)
            for row in rows:
                run_tag = row.get("Run tag", "").strip().strip("`")
                experiment_rows[run_tag] = {
                    "layers": normalize_layers(row.get("Engram layers", "")),
                    "memory_dim": row.get("Memory dim", "").strip().strip("`") or None,
                    "seed": row.get("Seed", "").strip().strip("`") or None,
                }
            continue

        if line.startswith("| Run tag | Train time"):
            _, rows, index = parse_markdown_table(lines, index)
            for row in rows:
                run_tag = row.get("Run tag", "").strip().strip("`")
                info = resolve_experiment_info(experiment_rows, run_tag)
                entry = {
                    "sweep": sweep_name,
                    "model": parse_run_tag_to_model(run_tag, sweep_name),
                    "layers": info.get("layers") or infer_layers_from_model(run_tag),
                    "memory_dim": info.get("memory_dim"),
                    "seed": info.get("seed"),
                    "base_core": to_float(row.get("CORE")),
                    "val_bpb": to_float(row.get("Val BPB") or row.get("Val bpb")),
                    "chatcore": None,
                    "arc_easy": None,
                    "arc_challenge": None,
                    "mmlu": None,
                    "gsm8k": None,
                    "humaneval": None,
                    "wall_clock": row.get("Train time (m)", "").strip().strip("`")
                    or None,
                    "source_summary": str(path.relative_to(REPO_DIR)),
                }
                results.append(entry)
            continue

        if line.startswith("| Run tag | ARC-Easy"):
            _, rows, index = parse_markdown_table(lines, index)
            for row in rows:
                run_tag = row.get("Run tag", "").strip().strip("`")
                target = None
                for entry in results:
                    if entry["model"] == parse_run_tag_to_model(run_tag, sweep_name):
                        target = entry
                        break
                if target is None:
                    continue
                target["arc_easy"] = to_float(row.get("ARC-Easy"))
                target["arc_challenge"] = to_float(row.get("ARC-Challenge"))
                target["mmlu"] = to_float(row.get("MMLU"))
                target["gsm8k"] = to_float(row.get("GSM8K"))
                target["humaneval"] = to_float(row.get("HumanEval"))
                if "ChatCORE" in row:
                    target["chatcore"] = to_float(row.get("ChatCORE"))
            continue

        match = re.search(r"- `([^`]+)` had `ChatCORE metric: ([0-9.]+)`", line)
        if match:
            run_tag = match.group(1)
            chatcore = float(match.group(2))
            for entry in results:
                if entry["model"] == parse_run_tag_to_model(run_tag, sweep_name):
                    entry["chatcore"] = chatcore
                    break
        index += 1

    return results


def parse_late_summary(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sweep_name = parse_sweep_name(path)
    results = []

    for index, line in enumerate(lines):
        if line.strip().startswith("| Topology |"):
            _, rows, _ = parse_markdown_table(lines, index)
            for row in rows:
                topology = normalize_layers(row.get("Topology", ""))
                if not topology:
                    continue
                compact = topology.replace(",", "")
                results.append(
                    {
                        "sweep": sweep_name,
                        "model": f"nano-engram-d24-sweep-{compact}-{sweep_name.replace('engram_sweep_', '')}",
                        "layers": topology,
                        "memory_dim": None,
                        "seed": None,
                        "base_core": to_float(row.get("BASE CORE")),
                        "val_bpb": to_float(row.get("val bpb") or row.get("Val BPB")),
                        "chatcore": to_float(row.get("ChatCORE")),
                        "arc_easy": to_float(row.get("ARC-Easy")),
                        "arc_challenge": to_float(row.get("ARC-Challenge")),
                        "mmlu": to_float(row.get("MMLU")),
                        "gsm8k": to_float(row.get("GSM8K")),
                        "humaneval": to_float(row.get("HumanEval")),
                        "wall_clock": (row.get("Wall clock") or "").strip().strip("`")
                        or None,
                        "source_summary": str(path.relative_to(REPO_DIR)),
                    }
                )
            break

    shared_settings_match = re.search(r"engram_memory_dim=?(?:`)?([0-9]+)(?:`)?", text)
    if not shared_settings_match:
        shared_settings_match = re.search(r"engram_memory_dim[^\n]*`([0-9]+)`", text)
    seed_match = re.search(r"engram_seed[^\n]*`([0-9]+)`", text)
    for entry in results:
        if shared_settings_match:
            entry["memory_dim"] = shared_settings_match.group(1)
        if seed_match:
            entry["seed"] = seed_match.group(1)
    return results


def parse_summary(path: Path):
    text = path.read_text(encoding="utf-8")
    if "| Topology |" in text:
        return parse_late_summary(path)
    return parse_early_summary(path)


def parse_report_file(report_path: Path):
    text = report_path.read_text(encoding="utf-8")
    relative_parts = report_path.relative_to(REPORTS_DIR).parts
    sweep_name = relative_parts[0]
    model = relative_parts[1]

    def extract(pattern, cast: Callable[[str], object] | None = float):
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        return cast(value) if cast is not None else value

    layers = infer_layers_from_model(model)
    wall_clock = extract(r"Total wall clock time: ([^\n]+)", cast=None)
    seed = extract(r'"seed":\s*([0-9]+)', cast=None)
    memory_dim = extract(r'"memory_dim":\s*([0-9]+)', cast=None)

    return {
        "sweep": sweep_name,
        "model": model,
        "layers": layers,
        "memory_dim": memory_dim,
        "seed": seed,
        "base_core": extract(r"- CORE metric: ([0-9.]+)"),
        "val_bpb": extract(r"- val bpb: ([0-9.]+)"),
        "chatcore": extract(r"- ChatCORE metric: ([0-9.]+)"),
        "arc_easy": extract(r"- ARC-Easy: ([0-9.]+)"),
        "arc_challenge": extract(r"- ARC-Challenge: ([0-9.]+)"),
        "mmlu": extract(r"- MMLU: ([0-9.]+)"),
        "gsm8k": extract(r"- GSM8K: ([0-9.]+)"),
        "humaneval": extract(r"- HumanEval: ([0-9.]+)"),
        "wall_clock": wall_clock,
        "source_summary": str(report_path.relative_to(REPO_DIR)),
    }


def format_metric(value):
    return "" if value is None else f"{value:.4f}"


def best_entry(entries, metric):
    valid = [entry for entry in entries if entry[metric] is not None]
    if not valid:
        return None
    reverse = metric != "val_bpb"
    return sorted(valid, key=lambda entry: entry[metric], reverse=reverse)[0]


def safe_mean(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def build_layer_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row["layers"]:
            continue
        grouped[row["layers"]].append(row)

    summary_rows = []
    for layers, entries in grouped.items():
        summary = {
            "layers": layers,
            "runs": len(entries),
        }
        for metric in METRIC_COLUMNS:
            summary[f"mean_{metric}"] = safe_mean([entry[metric] for entry in entries])
            best = best_entry(entries, metric)
            summary[f"best_{metric}"] = best[metric] if best else None
            summary[f"best_{metric}_sweep"] = best["sweep"] if best else ""
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            -(row["best_chatcore"] if row["best_chatcore"] is not None else -math.inf),
            -(
                row["best_base_core"]
                if row["best_base_core"] is not None
                else -math.inf
            ),
        )
    )
    return summary_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def style_axis(ax):
    ax.set_facecolor(CATPPUCCIN["base"])
    ax.tick_params(colors=CATPPUCCIN["text"])
    for spine in ax.spines.values():
        spine.set_color(CATPPUCCIN["surface1"])
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=CATPPUCCIN["subtext0"])


def plot_primary_metrics(layer_summary_rows):
    top_rows = layer_summary_rows[:12]
    labels = [row["layers"] for row in top_rows]
    metrics = [
        ("best_base_core", "BASE CORE", CATPPUCCIN["green"]),
        ("best_chatcore", "ChatCORE", CATPPUCCIN["blue"]),
        ("best_val_bpb", "val bpb", CATPPUCCIN["yellow"]),
    ]

    plt.style.use("dark_background")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor=CATPPUCCIN["base"])
    for ax, (metric_key, title, color) in zip(axs, metrics):
        values = [row[metric_key] for row in top_rows]
        ax.bar(range(len(labels)), values, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title, color=CATPPUCCIN["text"])
        style_axis(ax)
        if metric_key == "best_val_bpb":
            valid = [value for value in values if value is not None]
            if valid:
                pad = 0.002
                ax.set_ylim(max(valid) + pad, min(valid) - pad)
    fig.suptitle(
        "Engram progress by layer choice: primary metrics",
        color=CATPPUCCIN["text"],
        fontsize=16,
    )
    fig.tight_layout()
    output = OVERVIEW_DIR / "progress_by_layer_primary_metrics.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_secondary_heatmap(layer_summary_rows):
    top_rows = layer_summary_rows[:12]
    labels = [row["layers"] for row in top_rows]
    metric_keys = [
        "best_arc_easy",
        "best_arc_challenge",
        "best_mmlu",
        "best_gsm8k",
        "best_humaneval",
    ]
    metric_labels = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]
    data = [
        [row[key] if row[key] is not None else float("nan") for key in metric_keys]
        for row in top_rows
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=CATPPUCCIN["base"])
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Best secondary metrics by layer choice", color=CATPPUCCIN["text"])
    style_axis(ax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=CATPPUCCIN["text"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=CATPPUCCIN["text"])
    fig.tight_layout()
    output = OVERVIEW_DIR / "progress_by_layer_secondary_heatmap.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_position_effect(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row["layers"]:
            continue
        layers = [int(part) for part in row["layers"].split(",") if part]
        if not layers:
            continue
        grouped[len(layers)].append((max(layers), row["chatcore"], row["base_core"]))

    plt.style.use("dark_background")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor=CATPPUCCIN["base"])
    colors = {
        1: CATPPUCCIN["yellow"],
        2: CATPPUCCIN["green"],
        3: CATPPUCCIN["blue"],
        4: CATPPUCCIN["mauve"],
        5: CATPPUCCIN["red"],
        6: CATPPUCCIN["surface1"],
    }
    for count, points in sorted(grouped.items()):
        x = [point[0] for point in points]
        chat = [point[1] for point in points]
        base = [point[2] for point in points]
        axs[0].scatter(
            x,
            chat,
            label=f"{count} layers",
            color=colors.get(count, CATPPUCCIN["text"]),
            alpha=0.8,
        )
        axs[1].scatter(
            x,
            base,
            label=f"{count} layers",
            color=colors.get(count, CATPPUCCIN["text"]),
            alpha=0.8,
        )

    axs[0].set_title("ChatCORE vs terminal layer choice", color=CATPPUCCIN["text"])
    axs[0].set_xlabel("Latest engram layer")
    axs[0].set_ylabel("ChatCORE")
    axs[1].set_title("BASE CORE vs terminal layer choice", color=CATPPUCCIN["text"])
    axs[1].set_xlabel("Latest engram layer")
    axs[1].set_ylabel("BASE CORE")
    for ax in axs:
        style_axis(ax)
    axs[1].legend(frameon=False, labelcolor=CATPPUCCIN["text"])
    fig.tight_layout()
    output = OVERVIEW_DIR / "progress_by_terminal_layer_scatter.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_start_layer_effect(rows):
    filtered_rows = []
    for row in rows:
        if not row["layers"]:
            continue
        layers = [int(part) for part in row["layers"].split(",") if part]
        if len(layers) < 2:
            continue
        filtered_rows.append(
            (layers[0], layers[-1], len(layers), row["chatcore"], row["base_core"])
        )

    plt.style.use("dark_background")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor=CATPPUCCIN["base"])
    colors = {
        2: CATPPUCCIN["yellow"],
        3: CATPPUCCIN["green"],
        4: CATPPUCCIN["blue"],
        5: CATPPUCCIN["mauve"],
        6: CATPPUCCIN["red"],
    }

    grouped = defaultdict(list)
    for start, terminal, count, chat, base in filtered_rows:
        grouped[(count, terminal)].append((start, chat, base))

    for (count, terminal), points in sorted(grouped.items()):
        x = [point[0] for point in points]
        chat = [point[1] for point in points]
        base = [point[2] for point in points]
        label = f"{count} layers, tail {terminal}"
        color = colors.get(count, CATPPUCCIN["text"])
        axs[0].scatter(x, chat, label=label, color=color, alpha=0.75)
        axs[1].scatter(x, base, label=label, color=color, alpha=0.75)

    axs[0].set_title("ChatCORE vs start layer choice", color=CATPPUCCIN["text"])
    axs[0].set_xlabel("Earliest engram layer")
    axs[0].set_ylabel("ChatCORE")
    axs[1].set_title("BASE CORE vs start layer choice", color=CATPPUCCIN["text"])
    axs[1].set_xlabel("Earliest engram layer")
    axs[1].set_ylabel("BASE CORE")
    for ax in axs:
        style_axis(ax)
    axs[1].legend(frameon=False, labelcolor=CATPPUCCIN["text"], fontsize=8)
    fig.tight_layout()
    output = OVERVIEW_DIR / "progress_by_start_layer_scatter.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_summary_md(rows, best_per_sweep_rows, layer_summary_rows):
    best_chat = best_entry(rows, "chatcore")
    best_base = best_entry(rows, "base_core")
    best_val = best_entry(rows, "val_bpb")
    top_chat_rows = sorted(
        [row for row in layer_summary_rows if row["best_chatcore"] is not None],
        key=lambda row: row["best_chatcore"],
        reverse=True,
    )[:10]

    lines = [
        "# Engram Progress Overview",
        "This report summarizes archived nano-engram layer exploration across completed sweeps, organized by layer choice rather than experiment date.",
        "",
        "## Files",
        "- Per-run metrics CSV: `runs/reports/engram_progress_overview/all_runs_metrics.csv`",
        "- Best-per-sweep CSV: `runs/reports/engram_progress_overview/best_per_sweep.csv`",
        "- Per-layer summary CSV: `runs/reports/engram_progress_overview/best_per_layer.csv`",
        "- Primary metrics plot: `runs/reports/engram_progress_overview/progress_by_layer_primary_metrics.png`",
        "- Secondary metrics heatmap: `runs/reports/engram_progress_overview/progress_by_layer_secondary_heatmap.png`",
        "- Terminal-layer scatter: `runs/reports/engram_progress_overview/progress_by_terminal_layer_scatter.png`",
        "- Start-layer scatter: `runs/reports/engram_progress_overview/progress_by_start_layer_scatter.png`",
        "",
        "## Best-per-sweep table",
        "| Sweep | Best BASE CORE | BASE layers | Best ChatCORE | Chat layers | Best val bpb | val-bpb layers |",
        "|---|---:|---|---:|---|---:|---|",
    ]

    for row in best_per_sweep_rows:
        lines.append(
            f"| {row['sweep'].replace('engram_sweep_', '')} | {format_metric(row['best_base_core'])} | `{row['best_base_layers']}` | {format_metric(row['best_chatcore'])} | `{row['best_chat_layers']}` | {format_metric(row['best_val_bpb'])} | `{row['best_val_layers']}` |"
        )

    best_base_line = "- Best BASE CORE: unavailable"
    if best_base is not None:
        best_base_line = f"- Best BASE CORE: `{best_base['base_core']:.4f}` from `{best_base['layers']}` in `{best_base['sweep'].replace('engram_sweep_', '')}` (`{best_base['model']}`)"

    best_chat_line = "- Best ChatCORE: unavailable"
    if best_chat is not None:
        best_chat_line = f"- Best ChatCORE: `{best_chat['chatcore']:.4f}` from `{best_chat['layers']}` in `{best_chat['sweep'].replace('engram_sweep_', '')}` (`{best_chat['model']}`)"

    best_val_line = "- Best val bpb: unavailable"
    if best_val is not None:
        best_val_line = f"- Best val bpb: `{best_val['val_bpb']:.4f}` from `{best_val['layers']}` in `{best_val['sweep'].replace('engram_sweep_', '')}` (`{best_val['model']}`)"

    lines.extend(
        [
            "",
            "## All-time archived bests",
            best_base_line,
            best_chat_line,
            best_val_line,
            "",
            "## Top layer choices by best observed ChatCORE",
            "| Layers | Runs | Best ChatCORE | Best BASE CORE | Best val bpb | Best ARC-Challenge | Best MMLU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in top_chat_rows:
        lines.append(
            f"| `{row['layers']}` | {row['runs']} | {format_metric(row['best_chatcore'])} | {format_metric(row['best_base_core'])} | {format_metric(row['best_val_bpb'])} | {format_metric(row['best_arc_challenge'])} | {format_metric(row['best_mmlu'])} |"
        )

    lines.extend(
        [
            "",
            "## Layer-choice observations",
            "- The exploration frontier evolved from early/mid-layer patterns like `6,8`, `8,10`, and `4,8,10` toward sparse-late topologies centered on `8,12,*`.",
            "- Best downstream and best base-fit directions remain misaligned: dense or earlier stacks often help `BASE CORE` or `val bpb`, while sparse late anchors dominate `ChatCORE`.",
            "- The strongest late sparse results are still concentrated in the `8,12,16` to `8,12,19` neighborhood, with `8,12,17` currently the best observed ChatCORE point.",
            "- More recent start-layer sweeps broadened the map leftward: `3,12,17`, `4,12,17`, `5,12,17`, and `6,12,17` are all competitive enough to matter, even though they have not yet displaced `8,12,17` overall.",
            "- The latest structural sweeps show that removing the middle support layer hurts sharply, so the current frontier remains a three-layer sparse family rather than a simpler two-layer reduction.",
            "- Plots are organized by layer choice rather than date so repeated evaluation of the same topology and local neighborhood tradeoffs are easier to compare directly.",
            "",
            "## Figures",
            "",
            "### Primary metrics by layer choice",
            "![progress_by_layer_primary_metrics.png](progress_by_layer_primary_metrics.png)",
            "",
            "This grouped-bar figure compares the best observed `BASE CORE`, `ChatCORE`, and `val bpb` for the top layer choices. It is the right overview figure because those three metrics drive most selection decisions and often disagree.",
            "",
            "### Secondary metric heatmap",
            "![progress_by_layer_secondary_heatmap.png](progress_by_layer_secondary_heatmap.png)",
            "",
            "The heatmap is the right secondary figure because it compresses five downstream metrics into a single topology-by-metric view, making tradeoffs like ARC-Easy vs HumanEval easier to see.",
            "",
            "### Terminal-layer scatter",
            "![progress_by_terminal_layer_scatter.png](progress_by_terminal_layer_scatter.png)",
            "",
            "This scatter plot is the right structural figure because it shows how performance changes with the latest engram layer while coloring by topology size, which helps separate terminal-layer effects from simple layer-count effects.",
            "",
            "### Start-layer scatter",
            "![progress_by_start_layer_scatter.png](progress_by_start_layer_scatter.png)",
            "",
            "This scatter plot is the right complementary structural figure because it shows how performance changes with the earliest engram layer while grouping points by topology size and terminal layer. That makes the recent fixed-tail `12,17` start-layer exploration immediately visible without collapsing it into sweep chronology.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    summary_paths = sorted(REPORTS_DIR.glob("engram_sweep_*/summary.md"))
    all_rows = []
    for path in summary_paths:
        all_rows.extend(parse_summary(path))

    covered_pairs = {(row["sweep"], row["model"]) for row in all_rows}
    report_paths = sorted(REPORTS_DIR.glob("engram_sweep_*/**/report.md"))
    for report_path in report_paths:
        row = parse_report_file(report_path)
        key = (row["sweep"], row["model"])
        if key not in covered_pairs:
            all_rows.append(row)
            covered_pairs.add(key)

    deduped_rows = {}
    for row in all_rows:
        dedupe_key = (
            row["sweep"],
            row["layers"],
            row["base_core"],
            row["val_bpb"],
            row["chatcore"],
        )
        existing = deduped_rows.get(dedupe_key)
        if existing is None or existing["source_summary"].endswith("report.md"):
            deduped_rows[dedupe_key] = row

    all_rows = list(deduped_rows.values())

    all_rows.sort(key=lambda row: (row["sweep"], row["layers"], row["model"]))

    best_per_sweep_rows = []
    grouped_by_sweep = defaultdict(list)
    for row in all_rows:
        grouped_by_sweep[row["sweep"]].append(row)

    for sweep, entries in sorted(grouped_by_sweep.items()):
        best_base = best_entry(entries, "base_core")
        best_chat = best_entry(entries, "chatcore")
        best_val = best_entry(entries, "val_bpb")
        best_per_sweep_rows.append(
            {
                "sweep": sweep,
                "best_base_core": best_base["base_core"] if best_base else None,
                "best_base_model": best_base["model"] if best_base else "",
                "best_base_layers": best_base["layers"] if best_base else "",
                "best_chatcore": best_chat["chatcore"] if best_chat else None,
                "best_chat_model": best_chat["model"] if best_chat else "",
                "best_chat_layers": best_chat["layers"] if best_chat else "",
                "best_val_bpb": best_val["val_bpb"] if best_val else None,
                "best_val_model": best_val["model"] if best_val else "",
                "best_val_layers": best_val["layers"] if best_val else "",
            }
        )

    layer_summary_rows = build_layer_summary(all_rows)

    write_csv(
        OVERVIEW_DIR / "all_runs_metrics.csv",
        all_rows,
        [
            "sweep",
            "model",
            "layers",
            "memory_dim",
            "seed",
            "base_core",
            "val_bpb",
            "chatcore",
            "arc_easy",
            "arc_challenge",
            "mmlu",
            "gsm8k",
            "humaneval",
            "wall_clock",
            "source_summary",
        ],
    )
    write_csv(
        OVERVIEW_DIR / "best_per_sweep.csv",
        best_per_sweep_rows,
        [
            "sweep",
            "best_base_core",
            "best_base_model",
            "best_base_layers",
            "best_chatcore",
            "best_chat_model",
            "best_chat_layers",
            "best_val_bpb",
            "best_val_model",
            "best_val_layers",
        ],
    )
    write_csv(
        OVERVIEW_DIR / "best_per_layer.csv",
        layer_summary_rows,
        [
            "layers",
            "runs",
            "mean_base_core",
            "best_base_core",
            "best_base_core_sweep",
            "mean_val_bpb",
            "best_val_bpb",
            "best_val_bpb_sweep",
            "mean_chatcore",
            "best_chatcore",
            "best_chatcore_sweep",
            "mean_arc_easy",
            "best_arc_easy",
            "best_arc_easy_sweep",
            "mean_arc_challenge",
            "best_arc_challenge",
            "best_arc_challenge_sweep",
            "mean_mmlu",
            "best_mmlu",
            "best_mmlu_sweep",
            "mean_gsm8k",
            "best_gsm8k",
            "best_gsm8k_sweep",
            "mean_humaneval",
            "best_humaneval",
            "best_humaneval_sweep",
        ],
    )

    plot_primary_metrics(layer_summary_rows)
    plot_secondary_heatmap(layer_summary_rows)
    plot_position_effect(all_rows)
    plot_start_layer_effect(all_rows)

    summary_md = build_summary_md(all_rows, best_per_sweep_rows, layer_summary_rows)
    (OVERVIEW_DIR / "summary.md").write_text(summary_md, encoding="utf-8")


if __name__ == "__main__":
    main()
