"""Engram donor probe (21218, mhc-head-fix 20260507).

Probes the real-Engram checkpoint by swapping the Engram donor stream
(`engram_input_ids`) for matched / adversarial / unrelated text and measuring the
change in the target token's logit vs. the self (matched-to-prompt) donor.

Left panel : Δ best-target logit vs. self, grouped by donor type per case.
             All deltas are <= 0 -> donor text only lowers confidence.
Right panel: best-target logit per donor type (absolute), with the self value
             shown as a lavender star -> self is always highest.

Key finding (annotated): the target token stays rank-1 in EVERY case/variant,
so the donor path modulates confidence but never flips the prediction.

Palette: Catppuccin Latte.
Source: nanochat/runs/reports/engram_donor_probe_21218_mhcheadfix-20260507/results.csv
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

LATTE = {
    "base": "#eff1f5", "mantle": "#e6e9ef", "crust": "#dce0e8",
    "text": "#4c4f69", "subtext0": "#6c6f85", "surface1": "#bcc0cc",
    "lavender": "#7287fd", "teal": "#179299", "peach": "#fe640b",
    "maroon": "#e64553", "green": "#40a02b",
}

CASES = ["France capital", "Gold symbol", "Largest planet"]

# donor type -> color
DONORS = [("matched", LATTE["teal"]), ("adversarial", LATTE["peach"]), ("unrelated", LATTE["maroon"])]

# Δ best-target logit vs self  (case order matches CASES)
DELTA = {
    "matched":     [-0.1925, -0.3658, -1.2590],
    "adversarial": [-0.1925, -0.0508, -1.3732],
    "unrelated":   [-0.3419, -0.9318, -0.6168],
}
# absolute best-target logit
SELF_LOGIT = [15.7798, 15.4379, 16.0547]
ABS_LOGIT = {
    "matched":     [15.5873, 15.0721, 14.7957],
    "adversarial": [15.5873, 15.3871, 14.6814],
    "unrelated":   [15.4379, 14.5061, 15.4379],
}

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Δ target logit vs. self donor", "Absolute target logit (★ = self)"],
    horizontal_spacing=0.10,
)

for donor, color in DONORS:
    fig.add_trace(go.Bar(
        x=CASES, y=DELTA[donor], name=donor, marker_color=color, legendgroup=donor,
        marker_line=dict(color=LATTE["base"], width=0.8),
        hovertemplate=f"<b>{donor}</b><br>%{{x}}<br>Δlogit: %{{y:+.4f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=CASES, y=ABS_LOGIT[donor], name=donor, marker_color=color, legendgroup=donor,
        showlegend=False, marker_line=dict(color=LATTE["base"], width=0.8),
        hovertemplate=f"<b>{donor}</b><br>%{{x}}<br>logit: %{{y:.4f}}<extra></extra>",
    ), row=1, col=2)

# self reference: zero line (left) + stars (right)
fig.add_hline(y=0, line=dict(color=LATTE["lavender"], width=2), row=1, col=1)
fig.add_annotation(x=2.4, y=0, text="self (Δ=0)", showarrow=False, yanchor="bottom",
                   font=dict(size=10, color=LATTE["lavender"]), row=1, col=1)
fig.add_trace(go.Scatter(
    x=CASES, y=SELF_LOGIT, mode="markers", name="self donor",
    marker=dict(symbol="star", size=18, color=LATTE["lavender"],
                line=dict(color=LATTE["base"], width=1.2)),
    hovertemplate="self<br>%{x}<br>logit: %{y:.4f}<extra></extra>",
    legendgroup="self",
), row=1, col=2)

# rank-never-flips annotation
fig.add_annotation(
    xref="paper", yref="paper", x=0.5, y=1.11, showarrow=False,
    text="Target token stays <b>rank&nbsp;1</b> in all 3 cases × 3 donors (Δrank = 0) — donor modulates confidence, never the prediction",
    font=dict(size=16, color=LATTE["green"]),
)

fig.update_layout(
    barmode="group", bargap=0.28, bargroupgap=0.07,
    title=dict(text="Engram donor probe (real-Engram d24 checkpoint, step 7001)",
               font=dict(size=20, color=LATTE["text"]), x=0.5, xanchor="center"),
    paper_bgcolor=LATTE["base"], plot_bgcolor=LATTE["mantle"],
    font=dict(color=LATTE["text"], family="Inter, Helvetica, Arial, sans-serif"),
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    width=1100, height=600, margin=dict(t=120, l=60, r=30, b=90),
)
for r, c in [(1, 1), (1, 2)]:
    fig.update_xaxes(row=r, col=c, color=LATTE["text"], linecolor=LATTE["surface1"], showgrid=False)
    fig.update_yaxes(row=r, col=c, color=LATTE["text"], linecolor=LATTE["surface1"],
                     gridcolor=LATTE["crust"], zeroline=False)
fig.update_yaxes(row=1, col=1, title_text="Δ logit")
fig.update_yaxes(row=1, col=2, title_text="logit", range=[14, 16.4])
for ann in fig.layout.annotations[:2]:
    ann.font = dict(size=14, color=LATTE["text"])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(OUT_DIR, "donor_probe.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("wrote", html_path)
try:
    fig.write_image(os.path.join(OUT_DIR, "donor_probe.png"), scale=2)
    print("wrote PNG")
except Exception as e:
    print("PNG export skipped:", str(e)[:60])
