"""
Drawing helpers shared by the Streamlit demo and the GIF exporter.

A cascade (from outputs/demo/demo_cascades.json) looks like:
    {"thread_id", "event", "label", "text", "nodes": [{"id","parent","t","depth","followers",
     "verified"}...], "probs": {"5": 0.41, "10": 0.63, ...}}
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .plots import C, GRID, INK, INK2, SURFACE  # noqa: E402

WINDOW_ORDER = ["5", "10", "30", "60", "180", "full"]


# ---------------------------------------------------------------------------
# layout - computed once on the FULL cascade so nodes never jump during replay
# ---------------------------------------------------------------------------
def radial_layout(nodes):
    children = {}
    root = None
    for n in nodes:
        if n["parent"] is None:
            root = n["id"]
        else:
            children.setdefault(n["parent"], []).append(n["id"])
    for k in children:                                    # order children by time
        children[k].sort(key=lambda c: next(x["t"] for x in nodes if x["id"] == c))

    leaves = {}

    def count(v):
        kids = children.get(v, [])
        leaves[v] = 1 if not kids else sum(count(k) for k in kids)
        return leaves[v]

    count(root)
    pos = {root: (0.0, 0.0)}
    # first ring grows with the number of direct replies so busy cascades do not overlap
    r1 = max(1.0, len(children.get(root, [])) / 12.0)

    def radius(depth):
        return r1 + (depth - 1) * 1.0

    def place(v, a0, a1, depth):
        kids = children.get(v, [])
        a = a0
        for k in kids:
            span = (a1 - a0) * leaves[k] / leaves[v]
            mid = a + span / 2
            r = radius(depth + 1)
            pos[k] = (r * math.cos(mid), r * math.sin(mid))
            place(k, a, a + span, depth + 1)
            a += span

    place(root, 0, 2 * math.pi, 0)
    return pos


def visible(nodes, t_now):
    return [n for n in nodes if n["t"] <= t_now]


def risk_at(probs: dict, t_now: float):
    """Latest model estimate available at time t_now (None before the first window closes)."""
    if t_now == float("inf") and "full" in probs:
        return ("full", probs["full"])
    best = None
    for w in WINDOW_ORDER[:-1]:
        if t_now >= float(w) and w in probs:
            best = (w, probs[w])
    return best


# ---------------------------------------------------------------------------
def draw_cascade(ax, cascade, t_now, pos=None, title=None, highlight_recent=3.0):
    nodes = cascade["nodes"]
    pos = pos or radial_layout(nodes)
    vis = visible(nodes, t_now)
    vis_ids = {n["id"] for n in vis}
    ax.set_facecolor(SURFACE)
    # edges
    for n in vis:
        if n["parent"] is not None and n["parent"] in vis_ids:
            (x0, y0), (x1, y1) = pos[n["parent"]], pos[n["id"]]
            ax.plot([x0, x1], [y0, y1], color="#c9c8c3", lw=1.0, zorder=1)
    # nodes
    xs, ys, sizes, cols, edge = [], [], [], [], []
    for n in vis:
        x, y = pos[n["id"]]
        xs.append(x)
        ys.append(y)
        sizes.append(140 if n["parent"] is None else 18 + 9 * math.log1p(n.get("followers", 0)))
        if n["parent"] is None:
            cols.append(INK)
        elif n["t"] >= t_now - highlight_recent:
            cols.append(C["orange"])
        else:
            cols.append(C["blue"])
        edge.append(C["yellow"] if n.get("verified") else SURFACE)
    ax.scatter(xs, ys, s=sizes, c=cols, edgecolors=edge, linewidths=1.6, zorder=3)
    lim = max(1.5, max(math.hypot(*p) for p in pos.values()) + 0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    depth = max((n["depth"] for n in vis), default=0)
    clock = "complete cascade" if t_now == float("inf") else f"T + {t_now:5.1f} min"
    ax.text(0.0, -0.02, f"{clock}   replies: {len(vis) - 1}   depth: {depth}",
            transform=ax.transAxes, fontsize=10, color=INK2, family="monospace")
    return pos


def draw_risk(ax, cascade, t_now, reveal=False):
    """Risk meter (bar) + the history of estimates at each window."""
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    r = risk_at(cascade["probs"], t_now)
    ax.text(0, 0.95, "Misinformation risk", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 0.88, "structure only - the model never reads the text", fontsize=8.5, color=INK2)
    # bar
    ax.add_patch(plt.Rectangle((0, 0.72), 1, 0.08, color=GRID, lw=0))
    if r is None:
        ax.text(0, 0.62, "collecting first 5 minutes...", fontsize=11, color=INK2)
    else:
        w, p = r
        col = C["orange"] if p >= 0.5 else C["blue"]
        ax.add_patch(plt.Rectangle((0, 0.72), p, 0.08, color=col, lw=0))
        ax.text(0, 0.60, f"{p:.0%}", fontsize=26, fontweight="bold", color=INK)
        verdict = "FLAG for fact-check" if p >= 0.5 else "looks normal"
        when = "full cascade" if w == "full" else f"{w} min"
        ax.text(0.42, 0.62, f"{verdict}\n(estimate @ {when})", fontsize=9.5, color=INK2)
    ax.plot([0.5, 0.5], [0.70, 0.82], color=INK2, lw=1, ls=":")
    ax.text(0.5, 0.835, "flag at 50%", ha="center", fontsize=7.5, color=INK2)
    # history
    ax.text(0, 0.44, "estimate at each window", fontsize=9, color=INK2)
    for i, w in enumerate(WINDOW_ORDER[:-1]):
        y = 0.38 - i * 0.065
        done = (t_now >= float(w) or t_now == float("inf")) and w in cascade["probs"]
        lab = f"{w:>4} min"
        val = f"{cascade['probs'][w]:.0%}" if done else "  --"
        ax.text(0, y, lab, fontsize=9, color=INK2, family="monospace")
        ax.text(0.30, y, val, fontsize=9, color=INK if done else GRID, family="monospace")
        if done:
            p = cascade["probs"][w]
            ax.add_patch(plt.Rectangle((0.45, y - 0.005), 0.5 * p, 0.03,
                                       color=C["orange"] if p >= .5 else C["blue"], lw=0))
    if reveal:
        lab = cascade["label"].upper()
        col = C["orange"] if cascade["label"] == "false" else C["blue"]
        ax.add_patch(plt.Rectangle((0, -0.06), 1, 0.07, color=col, alpha=0.15, lw=0))
        ax.text(0.02, -0.035, f"Fact-check verdict: {lab}", fontsize=11, fontweight="bold", color=INK)


def frame(cascade, t_now, pos=None, reveal=False, title=None, figsize=(10.5, 5.6)):
    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    ax1 = fig.add_axes([0.01, 0.06, 0.60, 0.86])
    ax2 = fig.add_axes([0.64, 0.10, 0.34, 0.82])
    pos = draw_cascade(ax1, cascade, t_now, pos, title=title)
    draw_risk(ax2, cascade, t_now, reveal=reveal)
    return fig, pos


def race_frame(c1, c2, t_now, pos1, pos2, reveal=False, figsize=(12, 5.4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=SURFACE)
    for ax, c, p, name in ((axes[0], c1, pos1, "Story A"), (axes[1], c2, pos2, "Story B")):
        draw_cascade(ax, c, t_now, p, title=name)
        r = risk_at(c["probs"], t_now)
        txt = "risk: collecting..." if r is None else f"risk @ {r[0]}{'' if r[0] == 'full' else ' min'}: {r[1]:.0%}"
        col = INK2 if r is None else (C["orange"] if r[1] >= 0.5 else C["blue"])
        ax.text(1.0, 1.02, txt, transform=ax.transAxes, ha="right", fontsize=11,
                fontweight="bold", color=col)
        if reveal:
            ax.text(1.0, 0.95, f"verdict: {c['label'].upper()}", transform=ax.transAxes,
                    ha="right", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def replay_times(max_t=180.0, n_frames=60):
    """Non-linear clock: early minutes get more frames (that is where the story is)."""
    u = np.linspace(0, 1, n_frames)
    return list(np.round(max_t * u ** 2, 2))


def why_chart(explain_rows, title):
    """Horizontal 'where does this cascade sit' chart: true median - this cascade - false median."""
    n = len(explain_rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.65 * n + 1.3), facecolor=SURFACE)
    for i, r in enumerate(explain_rows[::-1]):
        lo, hi = min(r["true_median"], r["false_median"]), max(r["true_median"], r["false_median"])
        span = (hi - lo) or 1.0
        pad = 0.6 * span
        a, b = lo - pad, hi + pad

        def nx_(v):
            return (min(max(v, a), b) - a) / (b - a)
        ax.plot([0, 1], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([nx_(r["true_median"])], [i], s=90, color=C["blue"], zorder=3,
                   label="typical TRUE" if i == 0 else None)
        ax.scatter([nx_(r["false_median"])], [i], s=90, color=C["orange"], zorder=3,
                   label="typical FALSE" if i == 0 else None)
        ax.scatter([nx_(r["value"])], [i], s=150, marker="D", color=INK, zorder=4,
                   label="this cascade" if i == 0 else None)
        ax.text(1.03, i, f"{r['value']:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(n), [r["label"] for r in explain_rows[::-1]], fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.8, n - 0.4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
