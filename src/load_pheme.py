"""
Step 1 - Load the PHEME veracity dataset into two flat tables.

Expected layout (PHEME 9-event veracity release, figshare 6392078):

    all-rnr-annotated-threads/
        charliehebdo-all-rnr-threads/
            rumours/<thread_id>/
                source-tweets/<thread_id>.json
                reactions/<reply_id>.json ...
                annotation.json
                structure.json
            non-rumours/<thread_id>/ ...
        ferguson-all-rnr-threads/ ...

Outputs
    threads.csv : one row per conversation thread (event, rumour flag, veracity label)
    tweets.csv  : one row per tweet (source + reactions) with parent id, time, user metadata
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

TW_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"   # e.g. "Wed Jan 07 11:06:08 +0000 2015"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _visible(p: Path) -> bool:
    """Skip macOS metadata files such as ._xxx and .DS_Store."""
    return not p.name.startswith(".")


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_time(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, TW_TIME_FMT)
    except ValueError:
        return None


def convert_annotation(ann: dict) -> str | None:
    """
    Map PHEME annotation flags to a veracity label.
    Same logic as the dataset authors' reference code (Kochkina et al., 2018):
        misinformation=0, true=1 -> "true"
        misinformation=1, true=0 -> "false"
        misinformation=0, true=0 -> "unverified"
    """
    if "misinformation" in ann and "true" in ann:
        m, t = int(ann["misinformation"]), int(ann["true"])
        if m == 0 and t == 1:
            return "true"
        if m == 1 and t == 0:
            return "false"
        if m == 0 and t == 0:
            return "unverified"
        return None
    if "misinformation" in ann:
        return "false" if int(ann["misinformation"]) == 1 else "unverified"
    return None


def _flatten_structure(tree, parent=None, out=None):
    """structure.json is a nested dict {id: {child_id: {...}}}; leaves may be [] or {}."""
    if out is None:
        out = {}
    if isinstance(tree, dict):
        for node, children in tree.items():
            out[str(node)] = None if parent is None else str(parent)
            _flatten_structure(children, node, out)
    return out


def _tweet_row(tw: dict, thread_id: str, is_source: bool) -> dict:
    user = tw.get("user", {}) or {}
    reply_to = tw.get("in_reply_to_status_id_str") or tw.get("in_reply_to_status_id")
    return {
        "thread_id": thread_id,
        "tweet_id": str(tw.get("id_str") or tw.get("id")),
        "reply_to": None if reply_to in (None, "") else str(reply_to),
        "is_source": is_source,
        "created_at": _parse_time(tw.get("created_at")),
        "user_id": str(user.get("id_str") or user.get("id") or "unknown"),
        "followers": user.get("followers_count", 0) or 0,
        "friends": user.get("friends_count", 0) or 0,
        "statuses": user.get("statuses_count", 0) or 0,
        "verified": bool(user.get("verified", False)),
        "user_created_at": _parse_time(user.get("created_at")),
        "text": tw.get("text", "") or "",
    }


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------
def find_event_dirs(raw_dir: Path) -> list[Path]:
    dirs = sorted(p for p in Path(raw_dir).rglob("*-all-rnr-threads") if p.is_dir() and _visible(p))
    if not dirs:
        raise FileNotFoundError(
            f"No '*-all-rnr-threads' folders found under {raw_dir}.\n"
            "Download PHEME_veracity from figshare (article 6392078), extract it, and place the "
            "'all-rnr-annotated-threads' folder inside data/raw/  -- or run with --sample."
        )
    return dirs


def load_thread(thread_dir: Path, event: str, is_rumour: bool):
    thread_id = thread_dir.name
    src_files = [p for p in (thread_dir / "source-tweets").glob("*.json") if _visible(p)]
    if not src_files:
        return None, []
    source = _read_json(src_files[0])
    rows = [_tweet_row(source, thread_id, True)]

    react_dir = thread_dir / "reactions"
    if react_dir.exists():
        for p in react_dir.glob("*.json"):
            if _visible(p):
                try:
                    rows.append(_tweet_row(_read_json(p), thread_id, False))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    # parent links: prefer structure.json (official conversation tree), else in_reply_to.
    struct_path = thread_dir / "structure.json"
    if struct_path.exists():
        try:
            parents = _flatten_structure(_read_json(struct_path))
            for r in rows:
                if r["tweet_id"] in parents and not r["is_source"]:
                    r["reply_to"] = parents[r["tweet_id"]] or r["reply_to"]
        except json.JSONDecodeError:
            pass

    label = "nonrumour"
    if is_rumour:
        ann_path = thread_dir / "annotation.json"
        label = convert_annotation(_read_json(ann_path)) if ann_path.exists() else None

    thread = {
        "thread_id": thread_id,
        "event": event,
        "is_rumour": int(is_rumour),
        "veracity": label,                    # true / false / unverified / nonrumour
        "source_text": rows[0]["text"],
        "n_reactions": len(rows) - 1,
    }
    return thread, rows


def load_pheme(raw_dir: Path, verbose: bool = True):
    threads, tweets = [], []
    for ev_dir in find_event_dirs(raw_dir):
        event = ev_dir.name.replace("-all-rnr-threads", "")
        for sub, is_rumour in (("rumours", True), ("non-rumours", False)):
            folder = ev_dir / sub
            if not folder.exists():
                continue
            for tdir in sorted(p for p in folder.iterdir() if p.is_dir() and _visible(p)):
                th, rows = load_thread(tdir, event, is_rumour)
                if th is not None:
                    threads.append(th)
                    tweets.extend(rows)
        if verbose:
            n = sum(1 for t in threads if t["event"] == event)
            print(f"  loaded {event:<18} {n:>5} threads")

    threads_df = pd.DataFrame(threads)
    tweets_df = pd.DataFrame(tweets)

    # --- clean-up -----------------------------------------------------------
    tweets_df = tweets_df.dropna(subset=["created_at"]).drop_duplicates(["thread_id", "tweet_id"])
    src_time = (tweets_df[tweets_df.is_source].set_index("thread_id")["created_at"])
    tweets_df["t_min"] = (
        (tweets_df["created_at"] - tweets_df["thread_id"].map(src_time)).dt.total_seconds() / 60.0
    )
    tweets_df["t_min"] = tweets_df["t_min"].clip(lower=0)              # occasional clock skew
    tweets_df["account_age_days"] = (
        (tweets_df["created_at"] - tweets_df["user_created_at"]).dt.total_seconds() / 86400.0
    ).clip(lower=0)
    tweets_df = tweets_df[tweets_df["thread_id"].isin(src_time.index)]
    threads_df = threads_df[threads_df["thread_id"].isin(src_time.index)].reset_index(drop=True)
    return threads_df, tweets_df.reset_index(drop=True)


def summarise(threads_df: pd.DataFrame) -> pd.DataFrame:
    tab = (threads_df.groupby(["event", "veracity"]).size().unstack(fill_value=0))
    for c in ("true", "false", "unverified", "nonrumour"):
        if c not in tab:
            tab[c] = 0
    tab = tab[["true", "false", "unverified", "nonrumour"]]
    tab["rumours"] = tab[["true", "false", "unverified"]].sum(axis=1)
    tab["total"] = tab["rumours"] + tab["nonrumour"]
    tab.loc["TOTAL"] = tab.sum()
    return tab
