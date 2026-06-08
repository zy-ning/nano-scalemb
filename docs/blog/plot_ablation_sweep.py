"""Engram ablation sweep (21218, true-backbone MHC, mhc-head-fix 20260507).

Compares four variants on the same MHC backbone with Engram layers 2,12,18:
  - MHC baseline  (Engram off)
  - Real engram   (genuine n-gram memory)
  - Randomized    (Engram slots randomized)
  - Uniform       (Engram slots uniform)

Split by evaluation stage:
  Top row    : base model evaluation  -> CORE (higher better) + val bpb (lower better)
  Bottom row : SFT / chat evaluation  -> ChatCORE, ARC-E/C, MMLU, GSM8K, HumanEval

Palette: Catppuccin Latte. Lavender = real Engram (the hero), subtext = MHC baseline.
Source: nanochat/runs/reports/engram_ablation_sweep_21218_true_backbone_mhc-mhcheadfix-20260507
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

LATTE = {
    "base": "#eff1f5", "mantle": "#e6e9ef", "crust": "#dce0e8",
    "text": "#4c4f69", "subtext0": "#6c6f85", "surface1": "#bcc0cc",
    "lavender": "#7287fd", "peach": "#fe640b", "teal": "#179299",
}

# Variant -> color (order = legend/group order)
VARIANTS = [
    ("MHC baseline", LATTE["subtext0"]),
    ("Real engram",  LATTE["lavender"]),
    ("Randomized",   LATTE["peach"]),
    ("Uniform",      LATTE["teal"]),
]

# --- base model evaluation metrics ---
BASE_CORE = {"MHC baseline": 0.2626, "Real engram": 0.2707, "Randomized": 0.2520, "Uniform": 0.2534}
VALBPB    = {"MHC baseline": 0.7107, "Real engram": 0.7048, "Randomized": 0.7237, "Uniform": 0.7127}

# --- SFT / chat evaluation metrics (all higher = better) ---
SFT = [
    ("ChatCORE",  {"MHC baseline": 0.3765, "Real engram": 0.4095, "Randomized": 0.3853, "Uniform": 0.3946}),
    ("ARC-E",     {"MHC baseline": 0.6494, "Real engram": 0.6970, "Randomized": 0.6768, "Uniform": 0.6928}),
    ("ARC-C",     {"MHC baseline": 0.4957, "Real engram": 0.5631, "Randomized": 0.5265, "Uniform": 0.5316}),
    ("MMLU",      {"MHC baseline": 0.3660, "Real engram": 0.4047, "Randomized": 0.3829, "Uniform": 0.3862}),
    ("GSM8K",     {"MHC baseline": 0.1198, "Real engram": 0.1008, "Randomized": 0.1069, "Uniform": 0.1016}),
    ("HumanEval", {"MHC baseline": 0.1280, "Real engram": 0.1402, "Randomized": 0.1098, "Uniform": 0.1341}),
]

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}], [{"colspan": 2}, None]],
    row_heights=[0.42, 0.58], column_widths=[0.5, 0.5],
    subplot_titles=[
        "Base eval · CORE (higher better)",
        "Base eval · val bpb (lower better — axis inverted, up = better)",
        "SFT / chat evaluation (higher better)",
    ],
    horizontal_spacing=0.12, vertical_spacing=0.16,
)

sft_labels = [m[0] for m in SFT]
for variant, color in VARIANTS:
    common = dict(name=variant, marker_color=color, legendgroup=variant,
                  marker_line=dict(color=LATTE["base"], width=0.8),
                  textposition="outside", textfont=dict(size=9, color=LATTE["text"]),
                  cliponaxis=False)
    # base CORE
    fig.add_trace(go.Bar(
        x=["CORE"], y=[BASE_CORE[variant]], showlegend=False,
        texttemplate="%{y:.4f}",
        hovertemplate=f"<b>{variant}</b><br>CORE: %{{y:.4f}}<extra></extra>", **common,
    ), row=1, col=1)
    # val bpb (axis inverted below so lower = taller)
    fig.add_trace(go.Bar(
        x=["val bpb"], y=[VALBPB[variant]], showlegend=False,
        texttemplate="%{y:.4f}",
        hovertemplate=f"<b>{variant}</b><br>val bpb: %{{y:.4f}}<extra></extra>", **common,
    ), row=1, col=2)
    # SFT grouped bars (legend lives here)
    fig.add_trace(go.Bar(
        x=sft_labels, y=[m[1][variant] for m in SFT],
        texttemplate="%{y:.3f}",
        hovertemplate=f"<b>{variant}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>", **common,
    ), row=2, col=1)

fig.update_layout(
    barmode="group", bargap=0.28, bargroupgap=0.08,
    title=dict(text="Engram ablation sweep (d24, layers 2,12,18, learned MHC head)",
               font=dict(size=20, color=LATTE["text"]), x=0.5, xanchor="center"),
    paper_bgcolor=LATTE["base"], plot_bgcolor=LATTE["mantle"],
    font=dict(color=LATTE["text"], family="Inter, Helvetica, Arial, sans-serif"),
    legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center",
                bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    width=1180, height=820, margin=dict(t=80, l=60, r=30, b=90),
)
for r, c in [(1, 1), (1, 2), (2, 1)]:
    fig.update_xaxes(row=r, col=c, color=LATTE["text"], linecolor=LATTE["surface1"], showgrid=False)
    fig.update_yaxes(row=r, col=c, color=LATTE["text"], linecolor=LATTE["surface1"],
                     gridcolor=LATTE["crust"], zeroline=False)
fig.update_yaxes(row=1, col=1, range=[0.24, 0.285])         # zoom CORE (+ label headroom)
fig.update_yaxes(row=1, col=2, range=[0.736, 0.684])        # val bpb INVERTED: up = lower = better
fig.update_yaxes(row=2, col=1, range=[0, 0.80])             # SFT label headroom
for ann in fig.layout.annotations:
    ann.font = dict(size=14, color=LATTE["text"])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(OUT_DIR, "ablation_sweep.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("wrote", html_path)
try:
    fig.write_image(os.path.join(OUT_DIR, "ablation_sweep.png"), scale=2)
    print("wrote PNG")
except Exception as e:
    print("PNG export skipped:", str(e)[:60])
