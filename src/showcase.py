"""
Step 6 - Prepare everything the live demo needs, precomputed (so nothing can break on stage).

Exports outputs/demo/demo_cascades.json with:
  * a pool of showcase cascades (false + true) taken from Task A held-out predictions,
    i.e. the model was trained on OTHER events and never saw these cascades.
  * their reply trees, per-window risk estimates, and "why" explanations.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .features import FEATURE_DESCRIPTIONS

DEMO_WINDOWS = ["5", "10", "30", "60", "180", "full"]


def _nodes(tweets, tid):
    tw = tweets[tweets.thread_id == tid].sort_values("t_min")
    ids = set(tw.tweet_id)
    src = tw.loc[tw.is_source, "tweet_id"].iloc[0]
    depth = {src: 0}
    out = []
    for r in tw.itertuples(index=False):
        if r.is_source:
            parent = None
        else:
            parent = r.reply_to if r.reply_to in ids and r.reply_to != r.tweet_id else src
            if parent not in depth:          # parent appears later in time / missing: attach to root
                parent = src
        depth[r.tweet_id] = 0 if parent is None else depth[parent] + 1
        out.append({"id": r.tweet_id, "parent": parent, "t": round(float(r.t_min), 3),
                    "depth": depth[r.tweet_id], "followers": int(r.followers),
                    "verified": bool(r.verified)})
    return out


def build_demo_data(features, tweets, threads, preds, imp, out_path, main_model="RandomForest",
                    task="A_false_vs_true", n_each=8):
    p = preds[(preds.task == task) & (preds.model == main_model)]
    piv = p.pivot_table(index="thread_id", columns="window", values="prob")
    piv = piv[[w for w in DEMO_WINDOWS if w in piv.columns]]
    lab = threads.set_index("thread_id")
    size = lab["n_reactions"]
    cand = piv.join(size).join(lab[["veracity", "event"]])
    cand = cand[(cand.n_reactions >= 12) & (cand.n_reactions <= 90)]

    # good demo stories: correct early AND at the end; prefer confident ones
    fals = cand[(cand.veracity == "false") & (cand["10"] >= 0.5) & (cand["full"] >= 0.5)]
    fals = fals.assign(score=fals["10"] + fals["full"]).sort_values("score", ascending=False)
    tru = cand[(cand.veracity == "true") & (cand["10"] < 0.5) & (cand["full"] < 0.5)]
    tru = tru.assign(score=-(tru["10"] + tru["full"])).sort_values("score", ascending=False)
    # one honest "miss" so the demo can discuss limitations
    miss = cand[(cand.veracity == "false") & (cand["full"] < 0.5)].head(1)

    def pick(df, n):   # spread picks across events for variety
        out, seen = [], {}
        for tid, r in df.iterrows():
            if seen.get(r.event, 0) < max(2, n // 3):
                out.append(tid)
                seen[r.event] = seen.get(r.event, 0) + 1
            if len(out) == n:
                break
        return out

    chosen = pick(fals, n_each) + pick(tru, n_each) + list(miss.index)

    # explanations: top features per window vs class medians
    A = features[features.veracity.isin(["false", "true"])]
    med = A.groupby(["window", "veracity"]).median(numeric_only=True)
    top_by_w = {w: (imp[(imp.task == task) & (imp.window == w)]
                    .sort_values("importance", ascending=False)["feature"].head(6).tolist())
                for w in DEMO_WINDOWS}

    fidx = features.set_index(["thread_id", "window"])
    cascades = []
    for tid in chosen:
        expl = {}
        for w in DEMO_WINDOWS:
            rows = []
            for f in top_by_w.get(w, []):
                v = fidx.at[(tid, w), f]
                tm, fm = med.at[(w, "true"), f], med.at[(w, "false"), f]
                if pd.isna(v):
                    continue
                closer = "false" if abs(v - fm) < abs(v - tm) else "true"
                rows.append({"feature": f, "label": FEATURE_DESCRIPTIONS.get(f, f),
                             "value": float(v), "true_median": float(tm),
                             "false_median": float(fm), "leans": closer})
            expl[w] = rows
        cascades.append({
            "thread_id": tid,
            "event": lab.at[tid, "event"],
            "label": lab.at[tid, "veracity"],
            "text": lab.at[tid, "source_text"],
            "n_reactions": int(size[tid]),
            "kind": "miss" if tid in miss.index else ("false" if lab.at[tid, "veracity"] == "false" else "true"),
            "probs": {w: float(piv.at[tid, w]) for w in piv.columns if not np.isnan(piv.at[tid, w])},
            "nodes": _nodes(tweets, tid),
            "explain": expl,
        })

    out_path.write_text(json.dumps({"task": task, "model": main_model, "cascades": cascades},
                                   indent=1, default=str))
    return cascades
