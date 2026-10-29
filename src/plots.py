"""
Step 5 - Result charts (PNG, used in the report and the demo's Results page).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .features import FEATURE_DESCRIPTIONS  # noqa: E402

# --- palette (validated categorical order; text never wears series colour) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "gray": "#8a8984"}
CLASS_COL = {"true": C["blue"], "false": C["orange"], "unverified": C["aqua"],
             "nonrumour": C["yellow"]}
MODEL_COL = {"RandomForest": C["blue"], "GradBoost": C["orange"], "LogReg": C["aqua"]}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

TASK_TITLE = {"A_false_vs_true": "Task A - misinformation (false) vs true rumours",
              "B_rumour_vs_nonrumour": "Task B - rumour vs non-rumour"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110,
    })


_style()


def _xlabels(windows):
    return [f"{w} min" if w != "full" else "full" for w in windows]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def dataset_overview(threads, path):
    tab = threads.groupby(["event", "veracity"]).size().unstack(fill_value=0)
    order = ["false", "true", "unverified", "nonrumour"]
    tab = tab.reindex(columns=order, fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    left = np.zeros(len(tab))
    for c in order:
        ax.barh(tab.index, tab[c], left=left, color=CLASS_COL[c], label=c,
                edgecolor=SURFACE, linewidth=1.5, height=0.7)
        left += tab[c].to_numpy()
    for i, tot in enumerate(left):
        ax.text(tot + 15, i, f"{int(tot)}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("threads")
    ax.grid(axis="y", visible=False)
    ax.set_title("Threads per event and label")
    ax.legend(ncol=4, loc="lower right", fontsize=9)
    return _save(fig, path)


def cascade_growth(tweets, threads, path, horizon=180):
    lab = threads.set_index("thread_id")["veracity"]
    grid = np.arange(0, horizon + 1, 2)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for cls in ("false", "true", "nonrumour"):
        tids = lab[lab == cls].index
        sub = tweets[(~tweets["is_source"]) & tweets["thread_id"].isin(tids)]
        if not len(tids):
            continue
        counts = np.searchsorted(np.sort(sub["t_min"].to_numpy()), grid, side="right") / len(tids)
        ax.plot(grid, counts, lw=2, color=CLASS_COL[cls], label=cls)
        ax.text(grid[-1] + 2, counts[-1], cls, color=INK2, va="center", fontsize=9)
    for w in (10, 30, 60):
        ax.axvline(w, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.set_xlabel("minutes since source tweet")
    ax.set_ylabel("mean replies per thread")
    ax.set_title("How fast conversations grow (mean cumulative replies)")
    ax.set_xlim(0, horizon + 26)
    return _save(fig, path)


def performance_vs_time(metrics, text_base, task, windows, path, metric="auc", main_model="RandomForest"):
    m = metrics[(metrics.task == task) & (metrics.feature_set == "all")]
    xs = np.arange(len(windows))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for model, col in MODEL_COL.items():
        r = m[m.model == model].set_index("window").reindex([str(w) for w in windows])
        ax.plot(xs, r[metric], marker="o", ms=7, lw=2.6 if model == main_model else 1.6, color=col,
                label=f"{model} (main)" if model == main_model else model,
                markeredgecolor=SURFACE, markeredgewidth=2)
        if metric == "auc" and model == main_model:
            ax.fill_between(xs, r["auc_lo"], r["auc_hi"], color=col, alpha=0.12, lw=0)
    if text_base is not None:
        tb = text_base[text_base.task == task][metric].iloc[0]
        ax.axhline(tb, color=C["gray"], ls="--", lw=1.6, label="text baseline (TF-IDF)")
    ax.axhline(0.5 if metric == "auc" else np.nan, color=GRID, lw=1)
    ax.set_xticks(xs, _xlabels(windows))
    ax.set_xlabel("observation window after the source tweet")
    ax.set_ylabel("AUC-ROC" if metric == "auc" else "macro F1")
    ax.set_title(f"{'AUC' if metric == 'auc' else 'Macro-F1'} vs time  |  {TASK_TITLE[task]}")
    lo = np.nanmin(m[metric]) if len(m) else 0.5
    hi = np.nanmax(m["auc_hi"] if metric == "auc" else m[metric]) if len(m) else 1.0
    ax.set_ylim(min(0.45, lo - 0.05), min(1.0, hi + 0.08))
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def ablation(metrics, task, windows, path, main_model="RandomForest"):
    m = metrics[(metrics.task == task) & (metrics.model == main_model)]
    xs = np.arange(len(windows))
    lab = {"all": "structure + user metadata", "topology": "topology only (graph + timing)",
           "user_meta": "user metadata only"}
    cols = {"all": C["blue"], "topology": C["orange"], "user_meta": C["aqua"]}
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for fs in ("all", "topology", "user_meta"):
        r = m[m.feature_set == fs].set_index("window").reindex([str(w) for w in windows])
        ax.plot(xs, r["auc"], marker="o", ms=7, lw=2, color=cols[fs], label=lab[fs],
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xticks(xs, _xlabels(windows))
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("observation window")
    ax.set_title(f"Which signals carry the information?  |  {TASK_TITLE[task].split(' - ')[0]}")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def importance_heatmap(imp, task, windows, path, top=12):
    d = imp[imp.task == task].pivot_table(index="feature", columns="window", values="importance")
    d = d[[str(w) for w in windows if str(w) in d.columns]]
    d = d.loc[d.mean(axis=1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(d) + 1.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = max(d.to_numpy().max(), 1e-6)
    im = ax.imshow(d.clip(lower=0).to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(d.shape[1]), _xlabels(list(d.columns)))
    ax.set_yticks(range(d.shape[0]), d.index)
    ax.grid(False)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("drop in AUC when shuffled", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Permutation importance by window  |  {TASK_TITLE[task].split(' - ')[0]}")
    return _save(fig, path)


def feature_distributions(features, imp, task, path, window="30", k=4):
    if task == "A_false_vs_true":
        d = features[(features.window == window) & features.veracity.isin(["false", "true"])]
        groups = [("false", d.veracity == "false"), ("true", d.veracity == "true")]
    else:
        d = features[features.window == window]
        groups = [("rumour", d.is_rumour == 1), ("non-rumour", d.is_rumour == 0)]
    top = (imp[(imp.task == task) & (imp.window == window)]
           .sort_values("importance", ascending=False)["feature"].head(k).tolist())
    fig, axes = plt.subplots(1, len(top), figsize=(2.3 * len(top) + 0.6, 3.8))
    axes = np.atleast_1d(axes)
    cols = [C["orange"], C["blue"]]
    for ax, f in zip(axes, top):
        data = [d.loc[mask, f].dropna().to_numpy() for _, mask in groups]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks([1, 2], [g for g, _ in groups])
        ax.set_title(f, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Top features at {window} min: distribution by class", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    return _save(fig, path)


def time_to_flag_chart(ttf, windows, path):
    order = [str(w) for w in windows]
    xs = np.arange(len(windows))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    groups = [(1, "false rumours caught", C["orange"], -0.2), (0, "true rumours wrongly flagged", C["blue"], 0.2)]
    for yv, lab, col, off in groups:
        d = ttf[ttf["y"] == yv] if "y" in ttf else ttf
        if not len(d):
            continue
        counts = d["first_flag_window"].value_counts().reindex(order, fill_value=0)
        cum = (counts.cumsum() / len(d)).to_numpy()
        ax.bar(xs + off, cum, width=0.38, color=col, label=lab, edgecolor=SURFACE, linewidth=1.5)
        for x, v in zip(xs + off, cum):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(xs, _xlabels(windows))
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of threads flagged (cumulative)")
    ax.set_xlabel("flagged by this window (risk >= 0.5)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("How early are false rumours caught, and at what false-alarm cost?")
    return _save(fig, path)


def per_event_chart(per_event, task, path, windows=("30", "full")):
    d = per_event[(per_event.task == task) & per_event.window.isin(windows)]
    d = d.dropna(subset=["auc"])
    if d.empty:
        return None
    piv = d.pivot_table(index="event", columns="window", values="auc").reindex(columns=list(windows))
    piv = piv.sort_values(piv.columns[-1])
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(piv) + 1.6))
    cols = [C["aqua"], C["blue"]]
    h = 0.36
    for k, (w, c) in enumerate(zip(piv.columns, cols)):
        ax.barh(y + (k - 0.5) * h, piv[w], height=h, color=c, label=f"{w} min" if w != "full" else "full",
                edgecolor=SURFACE)
    ax.axvline(0.5, color=C["gray"], ls="--", lw=1)
    ax.set_yticks(y, piv.index)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC on the held-out event")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Generalisation to unseen events (leave-one-event-out)")
    return _save(fig, path)


def describe(feature):
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def pipeline_diagram(path):
    steps = [("1. Load PHEME", "9 events\nsource + replies\nlabels"),
             ("2. Build graphs", "reply cascade tree\nuser interaction\ngraph"),
             ("3. Truncate", "5 / 10 / 30 / 60 /\n180 min / full"),
             ("4. SNA features", "speed, shape,\ncentrality, Louvain,\nuser metadata"),
             ("5. Classify", "leave-one-event-out\nRF / GB / LogReg"),
             ("6. Evaluate", "AUC & F1 vs time\nvs text baseline")]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (h, b) in enumerate(steps):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((i + 0.06, 0.12), 0.84, 0.76,
                     boxstyle="round,pad=0.02,rounding_size=0.06", fc="#eef4fc", ec=C["blue"], lw=1.4))
        ax.text(i + 0.48, 0.74, h, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(i + 0.48, 0.40, b, ha="center", va="center", fontsize=8.8, color=INK2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.9, 0.5),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    return _save(fig, path)
