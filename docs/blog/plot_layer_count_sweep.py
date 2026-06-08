"""Plot Engram+mHC metrics vs. Engram layer count (1/2/3).

Four panels: train loss, val bpb, CORE, ChatCORE.

- Per-panel best at each layer count is marked with a lavender star and the
  stars are joined by a solid lavender "best frontier" line.
- Every config is plotted as a lighter-colored marker labelled with its layer set.
- Dashed light lines connect each config to its supersets one layer-count up
  (the derivation lattice, e.g. 3 -> 3,12 -> 3,12,17).

Palette: Catppuccin Latte.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Catppuccin Latte ---------------------------------------------------------
LATTE = {
    "base": "#eff1f5",
    "mantle": "#e6e9ef",
    "crust": "#dce0e8",
    "text": "#4c4f69",
    "subtext0": "#6c6f85",
    "surface2": "#acb0be",
    "surface1": "#bcc0cc",
    "overlay0": "#9ca0b0",
    "lavender": "#7287fd",
    "blue": "#1e66f5",
    "teal": "#179299",
    "green": "#40a02b",
    "yellow": "#df8e1d",
    "peach": "#fe640b",
    "mauve": "#8839ef",
    "sky": "#04a5e5",
    "pink": "#ea76cb",
    "maroon": "#e64553",
}

# --- Data ---------------------------------------------------------------------
# key -> (layer_count, {metric: value})
# metrics: loss (train CE, lower better), valbpb (lower better),
#          core (higher better), chatcore (higher better)
DATA = {
    "3":       (1, dict(loss=2.351923, valbpb=0.7093, core=0.2608, chatcore=0.3863)),
    "8":       (1, dict(loss=2.363374, valbpb=0.7126, core=0.2651, chatcore=0.3939)),
    "12":      (1, dict(loss=2.392471, valbpb=0.7214, core=0.2409, chatcore=0.3915)),
    "17":      (1, dict(loss=2.405106, valbpb=0.7255, core=0.2408, chatcore=0.3789)),
    "3,12":    (2, dict(loss=2.327539, valbpb=0.7050, core=0.2648, chatcore=0.3689)),
    "3,17":    (2, dict(loss=2.341675, valbpb=0.7095, core=0.2593, chatcore=0.4043)),
    "8,12":    (2, dict(loss=2.347721, valbpb=0.7114, core=0.2645, chatcore=0.3731)),
    "8,17":    (2, dict(loss=2.339507, valbpb=0.7083, core=0.2610, chatcore=0.3956)),
    "2,12,17": (3, dict(loss=2.371117, valbpb=0.7100, core=0.2587, chatcore=0.3936)),
    "3,12,17": (3, dict(loss=2.362306, valbpb=0.7079, core=0.2643, chatcore=0.4041)),
    "8,12,17": (3, dict(loss=2.361491, valbpb=0.7080, core=0.2690, chatcore=0.3880)),
    "9,12,17": (3, dict(loss=2.368065, valbpb=0.7100, core=0.2656, chatcore=0.3949)),
}

# mHC-only baseline (Engram disabled): horizontal reference line in every panel.
# Source: nanochat .../engram_ablation_sweep_21218_true_backbone_mhc-mhcheadfix-20260507
#         /nano-engram-d24-mhc-baseline-21218-ablation-true-paper-mhc-mhcheadfix-20260507
BASELINE = dict(loss=2.329425, valbpb=0.7107, core=0.2626, chatcore=0.3765)

# Lineage colors for the non-best ("lighter") markers.
LINEAGE_COLOR = {
    "3": LATTE["blue"], "3,12": LATTE["blue"], "3,17": LATTE["blue"], "3,12,17": LATTE["blue"],
    "8": LATTE["peach"], "8,12": LATTE["peach"], "8,17": LATTE["peach"], "8,12,17": LATTE["peach"],
    "12": LATTE["teal"], "17": LATTE["green"],
    "2,12,17": LATTE["yellow"], "9,12,17": LATTE["mauve"],
}

# Derivation lattice: subset -> superset, adjacent layer count only.
EDGES = [
    ("3", "3,12"), ("3", "3,17"), ("8", "8,12"), ("8", "8,17"),
    ("12", "3,12"), ("12", "8,12"), ("17", "3,17"), ("17", "8,17"),
    ("3,12", "3,12,17"), ("3,17", "3,12,17"),
    ("8,12", "8,12,17"), ("8,17", "8,12,17"),
]

METRICS = [
    ("loss", "Train loss (CE)", "min"),
    ("valbpb", "Validation bpb", "min"),
    ("core", "CORE", "max"),
    ("chatcore", "ChatCORE", "max"),
]


def best_at_count(metric: str, mode: str, count: int):
    pts = [(k, v[metric]) for k, (c, v) in DATA.items() if c == count]
    return (min if mode == "min" else max)(pts, key=lambda kv: kv[1])


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[m[1] for m in METRICS],
    horizontal_spacing=0.11, vertical_spacing=0.13,
)

POS = {(m[0]): (r, c) for m, (r, c) in zip(METRICS, [(1, 1), (1, 2), (2, 1), (2, 2)])}

for metric, title, mode in METRICS:
    row, col = POS[metric]
    first_panel = (row, col) == (1, 1)

    # --- derivation lattice (dashed, light) ---
    for sub, sup in EDGES:
        cs, vs = DATA[sub][0], DATA[sub][1][metric]
        cp, vp = DATA[sup][0], DATA[sup][1][metric]
        fig.add_trace(go.Scatter(
            x=[cs, cp],
            y=[vs, vp],
            mode="lines",
            line=dict(color=LATTE["surface2"], width=1.1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=col)

    # --- mHC-only baseline (Engram off): horizontal reference line ---
    fig.add_trace(go.Scatter(
        x=[0.6, 3.6], y=[BASELINE[metric]] * 2,
        mode="lines",
        line=dict(color=LATTE["subtext0"], width=1.8, dash="dash"),
        hovertemplate="mHC-only baseline<br>" + title + ": %{y:.4f}<extra></extra>",
        name="mHC-only baseline (no Engram)", legendgroup="baseline",
        showlegend=first_panel,
    ), row=row, col=col)
    fig.add_annotation(
        x=3.55, y=BASELINE[metric],
        text="mHC-only", showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9, color=LATTE["subtext0"]), row=row, col=col,
    )

    # --- all config markers (lighter, lineage-colored) + labels ---
    for key, (count, vals) in DATA.items():
        fig.add_trace(go.Scatter(
            x=[count],
            y=[vals[metric]],
            mode="markers+text",
            marker=dict(size=10, color=LINEAGE_COLOR[key],
                        line=dict(color=LATTE["base"], width=1.2), opacity=0.92),
            text=[key], textposition="middle right",
            textfont=dict(size=9, color=LATTE["subtext0"]),
            hovertemplate=f"layers {key}<br>{title}: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    # --- best frontier (lavender solid line + stars) ---
    bx, by, btxt = [], [], []
    for count in (1, 2, 3):
        k, v = best_at_count(metric, mode, count)
        bx.append(count); by.append(v); btxt.append(k)
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="lines+markers",
        line=dict(color=LATTE["lavender"], width=2.6),
        marker=dict(symbol="star", size=20, color=LATTE["lavender"],
                    line=dict(color=LATTE["base"], width=1.4)),
        hovertemplate="best @ %{x} layer(s)<br>" + title + ": %{y:.4f}<extra></extra>",
        customdata=btxt,
        name="Best frontier", legendgroup="best",
        showlegend=first_panel,
    ), row=row, col=col)

# --- axes ---------------------------------------------------------------------
for r in (1, 2):
    for c in (1, 2):
        fig.update_xaxes(
            tickmode="array", tickvals=[1, 2, 3],
            ticktext=["1 layer", "2 layers", "3 layers"],
            range=[0.6, 3.6], row=r, col=c,
            showgrid=True, gridcolor=LATTE["crust"], zeroline=False,
            linecolor=LATTE["surface1"], color=LATTE["text"],
            title_text="Engram layer count" if r == 2 else None,
        )
        fig.update_yaxes(
            row=r, col=c, showgrid=True, gridcolor=LATTE["crust"], zeroline=False,
            linecolor=LATTE["surface1"], color=LATTE["text"],
        )

fig.update_layout(
    title=dict(
        text="Engram + mHC: metrics vs. layer count (d24, learned MHC head)",
        font=dict(size=20, color=LATTE["text"]), x=0.5, xanchor="center",
    ),
    paper_bgcolor=LATTE["base"], plot_bgcolor=LATTE["mantle"],
    font=dict(color=LATTE["text"], family="Inter, Helvetica, Arial, sans-serif"),
    legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    width=1180, height=900, margin=dict(t=80, l=70, r=40, b=90),
)
for ann in fig.layout.annotations:
    ann.font = dict(size=14, color=LATTE["text"])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(OUT_DIR, "layer_count_sweep.html")
fig.write_html(html_path, include_plotlyjs="cdn")  # CDN JS -> small file (needs net)
print("wrote", html_path)

try:
    png_path = os.path.join(OUT_DIR, "layer_count_sweep.png")
    fig.write_image(png_path, scale=2)
    print("wrote", png_path)
except Exception as e:  # kaleido/chrome may be unavailable offline
    print("PNG export skipped:", e)
