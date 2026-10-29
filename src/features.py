"""
Step 3 - Feature extraction (network structure only - the tweet text is never used).

One feature row per (thread, window).  Features are grouped so we can run ablations:

    speed      : how fast the conversation grows
    shape      : geometry of the reply cascade tree
    user_graph : SNA metrics on the user interaction graph (centrality, communities)
    user_meta  : non-text node metadata (followers, account age, verified)
"""
from __future__ import annotations

import math
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from .graphs import build_cascade_tree, build_user_graph, node_depths, truncate

FULL_CAP_MIN = 1440.0          # cap for "time" features in the full-cascade window (24 h)

FEATURE_GROUPS = {
    "speed": [
        "n_reactions", "n_users", "reaction_rate", "time_first_reply",
        "mean_gap", "early_burst_frac",
    ],
    "shape": [
        "max_depth", "mean_depth", "max_breadth", "depth_breadth_ratio",
        "frac_direct_replies", "branching_factor", "leaf_frac", "structural_virality",
    ],
    "user_graph": [
        "ug_density", "src_degree_share", "ug_reciprocity", "ug_max_betweenness",
        "ug_mean_betweenness", "ug_mean_closeness", "n_communities", "modularity",
        "largest_comm_frac", "tweets_per_user",
    ],
    "user_meta": [
        "src_log_followers", "src_verified", "src_log_account_age",
        "rx_mean_log_followers", "rx_median_log_followers", "rx_frac_verified",
        "rx_frac_new_accounts", "rx_mean_log_account_age", "rx_mean_ff_ratio",
    ],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
TOPOLOGY_FEATURES = FEATURE_GROUPS["speed"] + FEATURE_GROUPS["shape"] + FEATURE_GROUPS["user_graph"]

FEATURE_DESCRIPTIONS = {
    "n_reactions": "Number of replies visible so far",
    "n_users": "Distinct users involved",
    "reaction_rate": "Replies per minute",
    "time_first_reply": "Minutes until the first reply",
    "mean_gap": "Mean minutes between consecutive replies",
    "early_burst_frac": "Share of replies in the first 20% of the window",
    "max_depth": "Longest reply chain (hops from source)",
    "mean_depth": "Average reply depth",
    "max_breadth": "Most replies at a single depth level",
    "depth_breadth_ratio": "Max depth / max breadth",
    "frac_direct_replies": "Share of replies aimed directly at the source",
    "branching_factor": "Mean children per non-leaf tweet",
    "leaf_frac": "Share of tweets nobody replied to",
    "structural_virality": "Mean pairwise distance in the cascade (Goel et al.)",
    "ug_density": "Density of the user interaction graph",
    "src_degree_share": "Share of users interacting with the source author (degree centrality)",
    "ug_reciprocity": "Share of user-user replies that are reciprocated",
    "ug_max_betweenness": "Max betweenness centrality (bridging user)",
    "ug_mean_betweenness": "Mean betweenness centrality",
    "ug_mean_closeness": "Mean closeness centrality",
    "n_communities": "Louvain communities in the user graph",
    "modularity": "Louvain modularity",
    "largest_comm_frac": "Share of users in the largest community",
    "tweets_per_user": "Tweets per participating user (repeat participation)",
    "src_log_followers": "log followers of the source author",
    "src_verified": "Source author is verified",
    "src_log_account_age": "log account age (days) of the source author",
    "rx_mean_log_followers": "Mean log followers of repliers",
    "rx_median_log_followers": "Median log followers of repliers",
    "rx_frac_verified": "Share of verified repliers",
    "rx_frac_new_accounts": "Share of repliers with accounts < 90 days old",
    "rx_mean_log_account_age": "Mean log account age of repliers",
    "rx_mean_ff_ratio": "Mean log(followers/friends) of repliers",
}


def _window_minutes(window, tw: pd.DataFrame) -> float:
    if window == "full":
        return float(min(max(tw["t_min"].max(), 1.0), FULL_CAP_MIN))
    return float(window)


def _speed(tw, win_min):
    rx = tw.loc[~tw["is_source"], "t_min"].sort_values().to_numpy()
    n = len(rx)
    first = float(rx[0]) if n else win_min
    if n >= 1:
        gaps = np.diff(np.concatenate([[0.0], rx]))
        mean_gap = float(gaps.mean())
    else:
        mean_gap = win_min
    burst = float((rx <= 0.2 * win_min).mean()) if n else 0.0
    return {
        "n_reactions": n,
        "n_users": tw["user_id"].nunique(),
        "reaction_rate": n / max(win_min, 1e-6),
        "time_first_reply": min(first, win_min),
        "mean_gap": min(mean_gap, win_min),
        "early_burst_frac": burst,
    }


def _shape(tree, src):
    n = tree.number_of_nodes()
    if n <= 1:
        return {k: 0.0 for k in FEATURE_GROUPS["shape"]} | {"leaf_frac": 1.0}
    depths = node_depths(tree, src)
    d = np.array([v for k, v in depths.items() if k != src])
    level_counts = np.bincount(d)
    max_depth = int(d.max())
    max_breadth = int(level_counts.max())
    internal = [v for v in tree if tree.out_degree(v) > 0]
    und = tree.to_undirected()
    sv = nx.average_shortest_path_length(und) if n <= 600 else float("nan")
    return {
        "max_depth": max_depth,
        "mean_depth": float(d.mean()),
        "max_breadth": max_breadth,
        "depth_breadth_ratio": max_depth / max_breadth,
        "frac_direct_replies": float((d == 1).mean()),
        "branching_factor": float(np.mean([tree.out_degree(v) for v in internal])),
        "leaf_frac": sum(1 for v in tree if tree.out_degree(v) == 0) / n,
        "structural_virality": sv,
    }


def _user_graph(ug, src_user, n_tweets, seed):
    n = ug.number_of_nodes()
    out = {"tweets_per_user": n_tweets / max(n, 1)}
    if n <= 1 or ug.number_of_edges() == 0:
        out.update({"ug_density": 0.0, "src_degree_share": 0.0, "ug_reciprocity": 0.0,
                    "ug_max_betweenness": 0.0, "ug_mean_betweenness": 0.0,
                    "ug_mean_closeness": 0.0, "n_communities": float(n), "modularity": 0.0,
                    "largest_comm_frac": 1.0 / max(n, 1)})
        return out
    und = ug.to_undirected()
    btw = nx.betweenness_centrality(und, normalized=True)
    clo = nx.closeness_centrality(und)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comms = nx.community.louvain_communities(und, weight="weight", seed=seed)
        mod = nx.community.modularity(und, comms, weight="weight")
    out.update({
        "ug_density": nx.density(und),
        "src_degree_share": und.degree(src_user) / (n - 1) if src_user in und else 0.0,
        "ug_reciprocity": nx.reciprocity(ug) or 0.0,
        "ug_max_betweenness": max(btw.values()),
        "ug_mean_betweenness": float(np.mean(list(btw.values()))),
        "ug_mean_closeness": float(np.mean(list(clo.values()))),
        "n_communities": float(len(comms)),
        "modularity": float(mod),
        "largest_comm_frac": max(len(c) for c in comms) / n,
    })
    return out


def _user_meta(tw):
    src = tw[tw["is_source"]].iloc[0]
    rx = tw[~tw["is_source"]].drop_duplicates("user_id")
    lf = np.log1p(rx["followers"].astype(float))
    la = np.log1p(rx["account_age_days"].astype(float))
    nan = float("nan")
    return {
        "src_log_followers": math.log1p(src["followers"]),
        "src_verified": float(src["verified"]),
        "src_log_account_age": math.log1p(src["account_age_days"]) if pd.notna(src["account_age_days"]) else nan,
        "rx_mean_log_followers": float(lf.mean()) if len(rx) else nan,
        "rx_median_log_followers": float(lf.median()) if len(rx) else nan,
        "rx_frac_verified": float(rx["verified"].mean()) if len(rx) else nan,
        "rx_frac_new_accounts": float((rx["account_age_days"] < 90).mean()) if len(rx) else nan,
        "rx_mean_log_account_age": float(la.mean()) if len(rx) else nan,
        "rx_mean_ff_ratio": float((lf - np.log1p(rx["friends"].astype(float))).mean()) if len(rx) else nan,
    }


def thread_features(thread_tweets: pd.DataFrame, window, seed: int = 42) -> dict:
    tw = truncate(thread_tweets, window)
    win_min = _window_minutes(window, thread_tweets)
    tree, src = build_cascade_tree(tw)
    ug = build_user_graph(tw, tree)
    src_user = tree.nodes[src]["user"]
    feats = {}
    feats.update(_speed(tw, win_min))
    feats.update(_shape(tree, src))
    feats.update(_user_graph(ug, src_user, len(tw), seed))
    feats.update(_user_meta(tw))
    return feats


def extract_all(threads_df: pd.DataFrame, tweets_df: pd.DataFrame, windows, seed=42, verbose=True):
    rows = []
    groups = {tid: g.sort_values("t_min") for tid, g in tweets_df.groupby("thread_id")}
    meta = threads_df.set_index("thread_id")
    total = len(groups)
    for i, (tid, g) in enumerate(groups.items(), 1):
        for w in windows:
            r = thread_features(g, w, seed)
            r.update({"thread_id": tid, "window": str(w),
                      "event": meta.at[tid, "event"],
                      "is_rumour": meta.at[tid, "is_rumour"],
                      "veracity": meta.at[tid, "veracity"]})
            rows.append(r)
        if verbose and (i % 500 == 0 or i == total):
            print(f"  features: {i}/{total} threads")
    cols = ["thread_id", "event", "window", "is_rumour", "veracity"] + ALL_FEATURES
    return pd.DataFrame(rows)[cols]
