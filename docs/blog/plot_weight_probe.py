"""Tier-1 weight-probe plot: what the Engram actually learned (CPU, weights only).

Reads docs/blog/weight_probe/weight_probe.json (produced by
scripts/engram_weight_probe.py on the real / randomize / uniform ablation
checkpoints) and renders three panels:

  1. Engram write strength  (value_proj RMS, init 0) per layer, per variant
  2. Engram read gate       (stream_key_proj norm / init) per layer, per variant
  3. Memory content         (embedding row-norm distribution, log-x) per variant

Story: real grows + uses the memory; randomize still *tries* to read frozen
noise; uniform shuts the read path off (value_proj -> 0, read gate < init)
because an information-free table is worth ignoring.

Palette: Catppuccin Latte (matches ablation_sweep: real=lavender, randomize=peach,
uniform=teal).
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(OUT_DIR, "weight_probe", "weight_probe.json"), encoding="utf-8") as f:
    DATA = json.load(f)


def find_record(key):
    for rec in DATA:
        if key in rec["model_tag"]:
            return rec
    raise KeyError(key)


fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "Engram write strength<br>(value_proj RMS, init 0)",
        "Engram read gate<br>(stream_key norm / init)",
        "Memory content<br>(embedding row-norm dist.)",
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

    # panel 1: write strength
    fig.add_trace(go.Bar(
        x=xlab, y=[L["value_proj_rms"] for L in rec["engram_layers"]],
        texttemplate="%{y:.3f}", showlegend=False,
        hovertemplate=f"<b>{label}</b><br>%{{x}} value_proj RMS: %{{y:.4f}}<extra></extra>",
        **common,
    ), row=1, col=1)

    # panel 2: read gate (ratio to init)
    fig.add_trace(go.Bar(
        x=xlab, y=[L.get("stream_key_proj_ratio") for L in rec["engram_layers"]],
        texttemplate="%{y:.2f}", showlegend=True,
        hovertemplate=f"<b>{label}</b><br>%{{x}} stream_key/init: %{{y:.2f}}<extra></extra>",
        **common,
    ), row=1, col=2)

    # panel 3: embedding row-norm distribution (use a mid Engram layer)
    L = rec["engram_layers"][len(rec["engram_layers"]) // 2]
    emb = L["embedding"]
    lo, hi, nb = emb["loghist_lo"], emb["loghist_hi"], emb["loghist_bins"]
    width = (hi - lo) / nb
    centers = [10 ** (lo + (i + 0.5) * width) for i in range(nb)]
    counts = emb["loghist_counts"]
    fig.add_trace(go.Scatter(
        x=centers, y=[c + 1 for c in counts],  # +1 so zeros are visible on log-y
        mode="lines", line=dict(color=color, width=2.4, shape="spline"),
        legendgroup=label, name=label, showlegend=False,
        hovertemplate=f"<b>{label}</b><br>row-norm≈%{{x:.1f}}<br>rows: %{{y}}<extra></extra>",
    ), row=1, col=3)

# init reference line on the read-gate panel
fig.add_hline(y=1.0, line=dict(color=LATTE["subtext0"], width=1.4, dash="dash"),
              row=1, col=2)
fig.add_annotation(x=2.4, y=1.0, text="init", showarrow=False, yanchor="bottom",
                   font=dict(size=9, color=LATTE["subtext0"]), row=1, col=2)
# init reference on memory panel (untouched row norm ~ sqrt(dim))
init_mean = find_record("real")["engram_layers"][0]["embedding"]["init_mean"]
fig.add_vline(x=init_mean, line=dict(color=LATTE["subtext0"], width=1.4, dash="dash"),
              row=1, col=3)
fig.add_annotation(x=init_mean, y=1, text="init  (untouched)", showarrow=False,
                   xanchor="left", yanchor="bottom",
                   font=dict(size=9, color=LATTE["subtext0"]), row=1, col=3)

fig.update_layout(
    barmode="group", bargap=0.28, bargroupgap=0.08,
    title=dict(text="What the Engram learned — weights only, CPU (d24, layers 2/12/18, ablation sweep)",
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
fig.update_yaxes(row=1, col=1, range=[0, 0.20], title_text="RMS")
fig.update_yaxes(row=1, col=2, range=[0, 5.4], title_text="ratio")
fig.update_xaxes(row=1, col=3, type="log", range=[0.78, 3.05],  # ~6 .. ~1100
                 title_text="embedding row norm (log)")
fig.update_yaxes(row=1, col=3, type="log", title_text="row count (+1)")
for ann in fig.layout.annotations:
    if ann.text in ("init", "init  (untouched)"):
        continue
    ann.font = dict(size=13, color=LATTE["text"])

html_path = os.path.join(OUT_DIR, "weight_probe.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("wrote", html_path)
try:
    fig.write_image(os.path.join(OUT_DIR, "weight_probe.png"), scale=2)
    print("wrote PNG")
except Exception as e:
    print("PNG export skipped:", str(e)[:60])
