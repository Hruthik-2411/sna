"""
Step 4 - Modelling and evaluation.

* Leave-one-event-out (LOEO) cross-validation: train on 8 events, test on the unseen 9th,
  repeat for every event, then pool the out-of-fold predictions.  This mimics the real
  situation - a brand-new breaking-news event the model has never seen.
* Structure-only classifiers at every time window (LogReg, Random Forest, Gradient Boosting).
* Ablation: topology-only vs user-metadata-only feature sets.
* Content baseline: TF-IDF + logistic regression on the source tweet text.
* Permutation importance (on held-out folds) to explain which signals matter when.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, FEATURE_GROUPS, TOPOLOGY_FEATURES

FEATURE_SETS = {
    "all": ALL_FEATURES,                          # structure + non-text user metadata
    "topology": TOPOLOGY_FEATURES,                # pure network structure + timing
    "user_meta": FEATURE_GROUPS["user_meta"],     # who is involved (no structure)
}


def make_models(seed=42):
    return {
        "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                             class_weight="balanced_subsample",
                                                             n_jobs=-1, random_state=seed)),
        "GradBoost": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                                    max_leaf_nodes=15, l2_regularization=1.0,
                                                    class_weight="balanced", random_state=seed),
    }


def task_frame(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Select rows and binary target for a task. y = 1 means 'positive / suspicious'."""
    if task == "A_false_vs_true":
        d = df[df["veracity"].isin(["false", "true"])].copy()
        y = (d["veracity"] == "false").astype(int).to_numpy()
    elif task == "B_rumour_vs_nonrumour":
        d = df.copy()
        y = d["is_rumour"].astype(int).to_numpy()
    else:
        raise ValueError(task)
    return d.reset_index(drop=True), y


def _metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "macro_f1": f1_score(y, yhat, average="macro"),
        "accuracy": accuracy_score(y, yhat),
    }


def bootstrap_auc(y, p, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            vals.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def loeo_predict(model, X: pd.DataFrame, y: np.ndarray, events: np.ndarray,
                 importance: bool = False, seed=42):
    """Leave-one-event-out out-of-fold probabilities (+ optional permutation importance)."""
    oof = np.full(len(y), np.nan)
    imp_sum, imp_w = np.zeros(X.shape[1]), 0.0
    for ev in np.unique(events):
        te = events == ev
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(model).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if importance and len(np.unique(y[te])) > 1 and te.sum() >= 30:
            Xte, yte = X[te], y[te]
            if len(yte) > 500:                               # subsample big folds for speed
                sub = np.random.default_rng(seed).choice(len(yte), 500, replace=False)
                Xte, yte = Xte.iloc[sub], yte[sub]
            if len(np.unique(yte)) < 2:
                continue
            r = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=3,
                                       random_state=seed, n_jobs=1)
            imp_sum += r.importances_mean * te.sum()
            imp_w += te.sum()
    imp = imp_sum / imp_w if imp_w else None
    return oof, imp


def run_structure_models(features: pd.DataFrame, task: str, windows, seed=42,
                         main_model="auto", verbose=True):
    """
    Pass 1: every model on all features at every window.
    The main model = the one with the best mean AUC over windows (or the one given).
    Pass 2: for the main model only - permutation importance, per-event scores, ablations.
    """
    d_all, _ = task_frame(features, task)
    metric_rows, pred_rows, imp_rows, event_rows = [], [], [], []
    models = make_models(seed)
    W = [str(x) for x in windows]

    def window_data(w):
        d = d_all[d_all["window"] == w].reset_index(drop=True)
        y = ((d["veracity"] == "false") if task == "A_false_vs_true"
             else (d["is_rumour"] == 1)).astype(int).to_numpy()
        return d, y, d["event"].to_numpy()

    def record(w, name, fset, y, oof, d, events):
        ok = ~np.isnan(oof)
        m = _metrics(y[ok], oof[ok])
        lo, hi = bootstrap_auc(y[ok], oof[ok])
        metric_rows.append({"task": task, "window": w, "model": name, "feature_set": fset,
                            **m, "auc_lo": lo, "auc_hi": hi, "n": int(ok.sum()),
                            "pos_rate": float(y[ok].mean())})
        if verbose:
            print(f"  [{task}] window={w:>4} {name:<12} {fset:<9} "
                  f"AUC={m['auc']:.3f}  F1={m['macro_f1']:.3f}")

    # ---- pass 1 -------------------------------------------------------------
    for w in W:
        d, y, events = window_data(w)
        for name, mdl in models.items():
            oof, _ = loeo_predict(mdl, d[FEATURE_SETS["all"]], y, events, seed=seed)
            record(w, name, "all", y, oof, d, events)
            pred_rows.append(pd.DataFrame({"task": task, "window": w, "model": name,
                                           "thread_id": d["thread_id"], "event": events,
                                           "y": y, "prob": oof}))
    if main_model == "auto":
        mm = pd.DataFrame(metric_rows)
        main_model = mm.groupby("model")["auc"].mean().idxmax()
    if verbose:
        print(f"  [{task}] main model = {main_model} (best mean AUC over windows)")

    # ---- pass 2 -------------------------------------------------------------
    for w in W:
        d, y, events = window_data(w)
        cols = FEATURE_SETS["all"]
        oof, imp = loeo_predict(models[main_model], d[cols], y, events, importance=True, seed=seed)
        ok = ~np.isnan(oof)
        if imp is not None:
            imp_rows += [{"task": task, "window": w, "feature": f, "importance": v}
                         for f, v in zip(cols, imp)]
        for ev in np.unique(events):
            te = (events == ev) & ok
            if te.sum() and len(np.unique(y[te])) > 1:
                event_rows.append({"task": task, "window": w, "event": ev,
                                   "n": int(te.sum()), **_metrics(y[te], oof[te])})
            elif te.sum():
                acc = float(((oof[te] >= .5).astype(int) == y[te]).mean())
                event_rows.append({"task": task, "window": w, "event": ev, "n": int(te.sum()),
                                   "auc": np.nan, "macro_f1": np.nan, "accuracy": acc})
        for fset in ("topology", "user_meta"):
            oof, _ = loeo_predict(models[main_model], d[FEATURE_SETS[fset]], y, events, seed=seed)
            record(w, main_model, fset, y, oof, d, events)

    return (pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True),
            pd.DataFrame(imp_rows), pd.DataFrame(event_rows), main_model)


def run_text_baseline(threads: pd.DataFrame, task: str, seed=42):
    """Content-based baseline: what the source tweet SAYS (TF-IDF + logistic regression)."""
    t = threads.copy()
    t["window"] = "full"
    d, y = task_frame(t, task)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                          lowercase=True),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0))
    events = d["event"].to_numpy()
    oof = np.full(len(y), np.nan)
    for ev in np.unique(events):
        te = events == ev
        if len(np.unique(y[~te])) < 2:
            continue
        m = clone(model).fit(d.loc[~te, "source_text"], y[~te])
        oof[te] = m.predict_proba(d.loc[te, "source_text"])[:, 1]
    ok = ~np.isnan(oof)
    met = _metrics(y[ok], oof[ok])
    lo, hi = bootstrap_auc(y[ok], oof[ok])
    return {"task": task, "model": "TF-IDF + LogReg (text)", **met, "auc_lo": lo, "auc_hi": hi,
            "n": int(ok.sum())}, pd.DataFrame({"thread_id": d["thread_id"], "y": y, "prob": oof})


def time_to_flag(preds: pd.DataFrame, windows, thr=0.5) -> pd.DataFrame:
    """For every thread: the earliest window at which its risk >= thr ('never' if none).
    For false rumours this is 'time to catch'; for true ones it is 'time to a false alarm'."""
    order = [str(w) for w in windows]
    p = preds.pivot_table(index=["thread_id", "y"], columns="window", values="prob")
    p = p[[c for c in order if c in p.columns]]
    out = []
    for (tid, y), row in p.iterrows():
        hit = next((c for c in p.columns if row[c] >= thr), "never")
        out.append({"thread_id": tid, "y": int(y), "first_flag_window": hit})
    return pd.DataFrame(out)
