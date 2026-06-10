"""Engram in-distribution token-flip probe (facts set, 21218, mhc-head-fix 20260507).

Successor to the donor probe. Holds the backbone prompt fixed and flips ONLY the
entity token in the Engram stream to a real, position-aligned, in-distribution
sentence (e.g. "The capital of France is" -> Engram reads "...Japan is"). Measures
whether the flip raises the *matching* answer (Tokyo) relative to the home answer
(Paris) -- a directional, causal content signal.

Left panel : mean directional signal by knowledge domain, grouped by checkpoint
             (real / randomize / uniform). real >> the two content-free controls;
             uniform is an exact 0 floor (identical table rows -> flip is a no-op).
Right panel: all 36 real cases sorted by directional signal, colored by domain, with
             the surface-controlled signal (flip vs. filler) overlaid as diamonds.
             The 8 sub-zero cases are the reversals; the diamonds show most are a
             tail-shift metric artifact or near-floor noise, leaving only a few
             genuinely-unlearned facts (e.g. element O/H, Greece->Athens).

Palette: Catppuccin Latte.
Source: runs/reports/engram_flip_probe_facts/{real,randomize,uniform}/results.csv
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

LATTE = {
    "base": "#eff1f5", "mantle": "#e6e9ef", "crust": "#dce0e8",
    "text": "#4c4f69", "subtext0": "#6c6f85", "surface1": "#bcc0cc",
    "lavender": "#7287fd", "teal": "#179299", "peach": "#fe640b",
    "maroon": "#e64553", "green": "#40a02b", "blue": "#1e66f5",
    "mauve": "#8839ef", "yellow": "#df8e1d", "sky": "#04a5e5",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(OUT_DIR, "..", "..", "runs", "reports", "engram_flip_probe_facts")

DOMAINS = ["capital", "language", "element", "continent", "planet"]
DOMAIN_COLOR = {
    "capital": LATTE["lavender"], "language": LATTE["blue"], "element": LATTE["green"],
    "continent": LATTE["yellow"], "planet": LATTE["mauve"],
}
CKPT_COLOR = {"real": LATTE["lavender"], "randomize": LATTE["peach"], "uniform": LATTE["teal"]}


def load(variant):
    path = os.path.join(REPORTS, variant, "results.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant_name"] != "flip":
                continue
            rows.append({
                "case": r["case_name"], "domain": r["tier"],
                "signal": float(r["directional_signal"]),
                "vs_filler": float(r["directional_signal_vs_filler"]),
                "home": r["home_target"], "target": r["variant_target"],
            })
    return rows


real, rand, uni = load("real"), load("randomize"), load("uniform")


def dom_mean(rows, key="signal"):
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r[key])
    out = {d: (sum(by[d]) / len(by[d]) if by[d] else 0.0) for d in DOMAINS}
    allv = [r[key] for r in rows]
    out["ALL"] = sum(allv) / len(allv) if allv else 0.0
    return out


cats = DOMAINS + ["ALL"]
labels = [d.capitalize() for d in DOMAINS] + ["ALL"]
means = {v: dom_mean(rows) for v, rows in [("real", real), ("randomize", rand), ("uniform", uni)]}

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.42, 0.58],
    subplot_titles=[
        "Mean directional signal by domain",
        "All 36 real cases (sorted) — bar: signal, ◆: surface-controlled",
    ],
    horizontal_spacing=0.11,
)

# ---- Left: grouped bars, real vs. randomize (uniform omitted: trivially all 0) ----
for v in ["real", "randomize"]:
    fig.add_trace(go.Bar(
        x=labels, y=[means[v][c] for c in cats], name=v, marker_color=CKPT_COLOR[v],
        legendgroup=v, legend="legend", marker_line=dict(color=LATTE["base"], width=0.8),
        hovertemplate=f"<b>{v}</b><br>%{{x}}<br>mean signal: %{{y:+.3f}}<extra></extra>",
    ), row=1, col=1)
fig.add_hline(y=0, line=dict(color=LATTE["surface1"], width=1), row=1, col=1)

# ---- Right: per-case sorted bars colored by domain, vs_filler diamonds ----
real_sorted = sorted(real, key=lambda r: r["signal"], reverse=True)
xs = list(range(len(real_sorted)))
seen = set()
for i, r in enumerate(real_sorted):
    show = r["domain"] not in seen
    seen.add(r["domain"])
    fig.add_trace(go.Bar(
        x=[i], y=[r["signal"]], name=r["domain"].capitalize() if show else None,
        marker_color=DOMAIN_COLOR[r["domain"]], legendgroup="d_" + r["domain"],
        legend="legend2", showlegend=show, width=0.78,
        marker_line=dict(color=LATTE["base"], width=0.4),
        customdata=[[r["case"], r["home"], r["target"], r["vs_filler"]]],
        hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]}"
                       "<br>signal: %{y:+.3f}<br>vs filler: %{customdata[3]:+.3f}<extra></extra>"),
    ), row=1, col=2)
fig.add_trace(go.Scatter(
    x=xs, y=[r["vs_filler"] for r in real_sorted], mode="markers",
    name="vs. filler (surface-controlled)", legendgroup="vsfiller", legend="legend2",
    marker=dict(symbol="diamond", size=7, color=LATTE["text"],
                line=dict(color=LATTE["base"], width=0.8)),
    hovertemplate="vs filler: %{y:+.3f}<extra></extra>",
), row=1, col=2)
fig.add_hline(y=0, line=dict(color=LATTE["surface1"], width=1), row=1, col=2)

# Shade the reversal region (cases with signal < 0)
n_rev = sum(1 for r in real_sorted if r["signal"] < 0)
if n_rev:
    fig.add_vrect(
        x0=len(real_sorted) - n_rev - 0.5, x1=len(real_sorted) - 0.5,
        fillcolor=LATTE["maroon"], opacity=0.07, line_width=0, row=1, col=2,
    )
    fig.add_annotation(
        x=len(real_sorted) - n_rev / 2 - 0.5, y=min(r["signal"] for r in real_sorted),
        text=f"{n_rev} reversals", showarrow=False, yshift=-12,
        font=dict(size=11, color=LATTE["maroon"]), row=1, col=2,
    )

real_mean = means["real"]["ALL"]
real_pos = sum(1 for r in real if r["signal"] > 0)
vf_mean = dom_mean(real, "vs_filler")["ALL"]
vf_pos = sum(1 for r in real if r["vs_filler"] > 0)
fig.add_annotation(
    xref="paper", yref="paper", x=0.5, y=1.13, showarrow=False,
    text=(f"Flipping the memory's entity token raises the matching answer: real "
          f"<b>+{real_mean:.2f}</b> logit, <b>{real_pos}/36</b> directional "
          f"(surface-controlled +{vf_mean:.2f}, {vf_pos}/36) — controls ≈ 0"),
    font=dict(size=15, color=LATTE["green"]),
)

fig.update_layout(
    barmode="group", bargap=0.28, bargroupgap=0.08,
    title=dict(text="Engram token-flip probe (real-Engram d24, step 7001, 36 cases × 5 domains)",
               font=dict(size=20, color=LATTE["text"]), x=0.5, xanchor="center"),
    paper_bgcolor=LATTE["base"], plot_bgcolor=LATTE["mantle"],
    font=dict(color=LATTE["text"], family="Inter, Helvetica, Arial, sans-serif"),
    legend=dict(orientation="h", y=-0.17, x=0.19, xanchor="center", yanchor="top",
                title=dict(text="checkpoint", font=dict(size=11)),
                bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    legend2=dict(orientation="h", y=-0.17, x=0.74, xanchor="center", yanchor="top",
                 title=dict(text="domain", font=dict(size=11)),
                 bgcolor=LATTE["mantle"], bordercolor=LATTE["surface1"], borderwidth=1),
    width=1180, height=620, margin=dict(t=130, l=60, r=30, b=120),
)
fig.update_xaxes(row=1, col=1, color=LATTE["text"], linecolor=LATTE["surface1"], showgrid=False)
fig.update_yaxes(row=1, col=1, title_text="mean Δ logit (matched − home)", color=LATTE["text"],
                 linecolor=LATTE["surface1"], gridcolor=LATTE["crust"], zeroline=False)
fig.update_xaxes(row=1, col=2, title_text="case rank (by signal)", color=LATTE["text"],
                 linecolor=LATTE["surface1"], showgrid=False, showticklabels=False)
fig.update_yaxes(row=1, col=2, title_text="directional signal (logit)", color=LATTE["text"],
                 linecolor=LATTE["surface1"], gridcolor=LATTE["crust"], zeroline=False)
for ann in fig.layout.annotations[:2]:
    ann.font = dict(size=14, color=LATTE["text"])

html_path = os.path.join(OUT_DIR, "flip_probe.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("wrote", html_path)
try:
    fig.write_image(os.path.join(OUT_DIR, "flip_probe.png"), scale=2)
    print("wrote PNG")
except Exception as e:
    print("PNG export skipped:", str(e)[:60])
