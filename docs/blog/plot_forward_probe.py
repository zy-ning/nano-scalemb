"""Tier-2 forward-pass probe plot: what the Engram does at inference.

Companion to plot_weight_probe.py (Tier-1, weights only). Reads
docs/blog/forward_probe/forward_probe.json (produced by
scripts/engram_forward_probe.py on the real / randomize / uniform ablation
checkpoints) and renders three panels measured on a fixed batch of real
validation tokens:

  1. Engram read gate      (per-token gate distribution per layer, as box plots)
  2. Gate selectivity      (across-token std of the gate -> responds to content?)
  3. Engram contribution   (per-stream output RMS actually written, log-y)

Story (the inference-time counterpart of the weight probe): real opens a
graded gate and writes a large contribution; randomize slams the gate to its
0/1 rails (wide p10..p90 whiskers) yet writes almost nothing, because the
frozen-noise payload it reads has tiny norm; uniform shuts the gate at depth
and also writes ~nothing -- an information-free table is worth ignoring. The
contribution RMS (panel 3) is the headline: real writes ~25x more than either
ablation, mirroring Tier-1's value_proj / memory-content collapse at inference.

Palette: Catppuccin Latte (matches weight_probe / ablation_sweep:
real=lavender, randomize=peach, uniform=teal).
"""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

LATTE = {
    "base": "#eff1f5", "mantle": "#e6e9ef", "crust": "#dce0e8",
    "text": "#4c4f69", "subtext0": "#6c6f85", "surface1": "#bcc0cc",
    "lavender": "#7287fd", "peach": "#fe640b", "teal": "#179299",
}

# checkpoint model_tag substring -> (display label, color)
VARIANTS = [
    ("real",      "Real engram", LATTE["lavender"]),
    ("randomize", "Randomized",  LATTE["peach"]),
    ("uniform",   "Uniform",     LATTE["teal"]),
]

def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(OUT_DIR, "forward_probe", "forward_probe.json"), encoding="utf-8") as f:
    DATA = json.load(f)


def find_record(key):
    for rec in DATA:
        # match the variant word but not the mhc-baseline record
        if key in rec["model_tag"] and rec["engram_layers"]:
            return rec
    raise KeyError(key)


fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "Engram read gate<br>(gate distribution: box = Q1–Q3, ─ mean)",
        "Gate selectivity<br>(across-token std)",
        "Engram contribution<br>(output RMS, log)",
    ],
    horizontal_spacing=0.075,
)

for key, label, color in VARIANTS:
    rec = find_record(key)
    layers = [L["layer"] for L in rec["engram_layers"]]
    xlab = [f"L{l}" for l in layers]
    common = dict(name=label, marker_color=color, legendgroup=label,
                  marker_line=dict(color=LATTE["base"], width=0.8),
                  textposition="outside", textfont=dict(size=9, color=LATTE["text"]),
                  cliponaxis=False)

    # panel 1: gate distribution as grouped box plots (one trace per variant,
    # spanning all layers via the x categories). Box = Q1..Q3 + median, whiskers
    # to the data range, dashed line = mean. Reads the spread cleanly: real sits
    # low with a modest box, randomize's box stretches to both rails (0..1),
    # uniform collapses to ~0.
    bx, by = [], []
    for L in rec["engram_layers"]:
        s = L["gate_samples"]
        bx.extend([f"L{L['layer']}"] * len(s))
        by.extend(s)
    fig.add_trace(go.Box(
        x=bx, y=by, name=label, legendgroup=label,
        line_color=color, fillcolor=rgba(color, 0.45), line_width=1.2,
        boxmean=True, whiskerwidth=0.6, boxpoints=False, showlegend=False,
        hovertemplate=f"<b>{label}</b><br>%{{x}} gate: %{{y}}<extra></extra>",
    ), row=1, col=1)

    # panel 2: gate selectivity (across-token std)
    fig.add_trace(go.Bar(
        x=xlab, y=[L["gate_selectivity"] for L in rec["engram_layers"]],
        texttemplate="%{y:.2f}", showlegend=True,
        hovertemplate=f"<b>{label}</b><br>%{{x}} selectivity: %{{y:.3f}}<extra></extra>",
        **common,
    ), row=1, col=2)

    # panel 3: contribution RMS (log-y so the uniform collapse is visible)
    fig.add_trace(go.Bar(
        x=xlab, y=[L["engram_out_rms"] for L in rec["engram_layers"]],
        texttemplate="%{y:.2g}", showlegend=False,
        hovertemplate=f"<b>{label}</b><br>%{{x}} out RMS: %{{y:.3f}}<extra></extra>",
        **common,
    ), row=1, col=3)

# init reference on the gate panel: at init keys are random, query norm=1, so the
# dot product is ~0 and the gate sits at sigmoid(0) = 0.5.
fig.add_hline(y=0.5, line=dict(color=LATTE["subtext0"], width=1.4, dash="dash"),
              row=1, col=1)
fig.add_annotation(x=2.4, y=0.5, text="init ≈ 0.5", showarrow=False, yanchor="bottom",
                   font=dict(size=9, color=LATTE["subtext0"]), row=1, col=1)

fig.update_layout(
    barmode="group", bargap=0.28, bargroupgap=0.08, boxmode="group",
    title=dict(text="What the Engram does at inference — forward pass on real text "
                    "(d24, layers 2/12/18, ablation sweep)",
               font=dict(size=19, color=LATTE["text"]), x=0.5, xanchor="center"),
    paper_bgcolor=LATTE["base"], plot_bgcolor=LATTE["mantle"],
    font=dict(color=LATTE["text"], family="Inter, Helvetica, Arial, sans-serif"),
    legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    width=1280, height=560, margin=dict(t=90, l=60, r=30, b=80),
)
for c in (1, 2, 3):
    fig.update_xaxes(row=1, col=c, color=LATTE["text"], linecolor=LATTE["surface1"], showgrid=False)
    fig.update_yaxes(row=1, col=c, color=LATTE["text"], linecolor=LATTE["surface1"],
                     gridcolor=LATTE["crust"], zeroline=False)
fig.update_yaxes(row=1, col=1, range=[0, 1.0], title_text="gate")
fig.update_yaxes(row=1, col=2, rangemode="tozero", title_text="std")
fig.update_yaxes(row=1, col=3, type="log", title_text="RMS (log)")
for ann in fig.layout.annotations:
    if ann.text == "init ≈ 0.5":
        continue
    ann.font = dict(size=13, color=LATTE["text"])

html_path = os.path.join(OUT_DIR, "forward_probe.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("wrote", html_path)
try:
    fig.write_image(os.path.join(OUT_DIR, "forward_probe.png"), scale=2)
    print("wrote PNG")
except Exception as e:
    print("PNG export skipped:", str(e)[:60])
