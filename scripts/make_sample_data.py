"""
Generate SYNTHETIC test data in the exact PHEME veracity folder format.

Purpose: lets you test the whole pipeline (and rehearse the demo) before / without
downloading PHEME.  Results on this data are NOT research results.

Class sizes per event mirror the real PHEME 9-event release.  Cascades are simulated
with mild, noisy differences between false, true, unverified and non-rumour threads.

    python scripts/make_sample_data.py            # -> data/sample/all-rnr-annotated-threads
"""
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SAMPLE_DIR  # noqa: E402

# (true, false, unverified, non-rumour) per event - same as real PHEME
EVENTS = {
    "charliehebdo": (193, 116, 149, 1621),
    "ebola-essien": (0, 14, 0, 0),
    "ferguson": (10, 8, 266, 859),
    "germanwings-crash": (94, 111, 33, 231),
    "gurlitt": (59, 0, 2, 77),
    "ottawashooting": (329, 72, 69, 420),
    "prince-toronto": (0, 222, 7, 4),
    "putinmissing": (0, 9, 117, 112),
    "sydneysiege": (382, 86, 54, 699),
}
EVENT_START = {
    "charliehebdo": datetime(2015, 1, 7, 11, 0, tzinfo=timezone.utc),
    "ebola-essien": datetime(2014, 10, 12, 9, 0, tzinfo=timezone.utc),
    "ferguson": datetime(2014, 8, 9, 20, 0, tzinfo=timezone.utc),
    "germanwings-crash": datetime(2015, 3, 24, 10, 0, tzinfo=timezone.utc),
    "gurlitt": datetime(2014, 11, 24, 8, 0, tzinfo=timezone.utc),
    "ottawashooting": datetime(2014, 10, 22, 14, 0, tzinfo=timezone.utc),
    "prince-toronto": datetime(2014, 11, 3, 15, 0, tzinfo=timezone.utc),
    "putinmissing": datetime(2015, 3, 12, 9, 0, tzinfo=timezone.utc),
    "sydneysiege": datetime(2014, 12, 15, 0, 0, tzinfo=timezone.utc),
}

# Simulation parameters per class.  Differences are deliberately modest and noisy.
#              size_mu  time_scale  p_deep  fol_mu  new_acc  verified_src  text cue words
PARAMS = {
    "false":      (2.32, 32.0, 0.36, 6.0, 0.15, 0.17, ["BREAKING", "unconfirmed", "reports", "allegedly", "!!"]),
    "true":       (2.26, 40.0, 0.30, 6.3, 0.11, 0.25, ["confirmed", "police", "statement", "official", "says"]),
    "unverified": (2.30, 36.0, 0.33, 6.1, 0.13, 0.20, ["reportedly", "claims", "sources", "possible", "rumour"]),
    "nonrumour":  (2.18, 44.0, 0.28, 6.4, 0.10, 0.28, ["update", "live", "news", "watch", "today"]),
}
TOPIC = {
    "charliehebdo": ["Charlie Hebdo", "Paris", "gunmen", "magazine"],
    "ebola-essien": ["Essien", "Ebola", "Milan", "footballer"],
    "ferguson": ["Ferguson", "Michael Brown", "police", "protest"],
    "germanwings-crash": ["Germanwings", "Alps", "co-pilot", "flight"],
    "gurlitt": ["Gurlitt", "Bern museum", "art collection", "bequest"],
    "ottawashooting": ["Ottawa", "Parliament Hill", "soldier", "shooting"],
    "prince-toronto": ["Prince", "Toronto", "concert", "Massey Hall"],
    "putinmissing": ["Putin", "Kremlin", "missing", "health"],
    "sydneysiege": ["Sydney", "Lindt cafe", "hostages", "Martin Place"],
}
FMT = "%a %b %d %H:%M:%S +0000 %Y"


def user_obj(rng, uid, fol_mu, new_rate, verified_p, now):
    new = rng.random() < new_rate
    age_days = rng.integers(3, 90) if new else rng.integers(120, 2900)
    followers = int(np.exp(rng.normal(fol_mu + (0 if not new else -1.2), 1.6)))
    return {
        "id": uid, "id_str": str(uid), "screen_name": f"user{uid}",
        "followers_count": followers,
        "friends_count": int(np.exp(rng.normal(5.8, 1.1))),
        "statuses_count": int(np.exp(rng.normal(8.0, 1.5))),
        "verified": bool(rng.random() < verified_p),
        "created_at": (now - timedelta(days=int(age_days))).strftime(FMT),
    }


def simulate_thread(rng, cls, event, tid_base, user_pool):
    size_mu, tscale, p_deep, fol_mu, new_rate, ver_src, cues = PARAMS[cls]
    # event-level noise so events differ (as in real data)
    ev_shift = (sum(map(ord, event)) % 7 - 3) * 0.04
    n_rx = int(min(max(np.exp(rng.normal(size_mu + ev_shift, 0.85)) - 1, 0), 320))
    t0 = EVENT_START[event] + timedelta(minutes=float(rng.uniform(0, 72 * 60)))

    src_uid = int(rng.choice(user_pool))
    src_user = user_obj(rng, src_uid, fol_mu + 1.0, new_rate * 0.6, ver_src, t0)
    topic = TOPIC[event]
    # text: topic words + a noisy class cue (gives the text baseline something to learn)
    words = [rng.choice(topic), rng.choice(topic)]
    cue_pool = cues if rng.random() < 0.30 else PARAMS[rng.choice(list(PARAMS))][6]
    words += list(rng.choice(cue_pool, size=2))
    text = " ".join(words) + f" #{topic[0].replace(' ', '')}"
    source = {"id": tid_base, "id_str": str(tid_base), "text": text,
              "created_at": t0.strftime(FMT), "in_reply_to_status_id": None,
              "in_reply_to_status_id_str": None, "user": src_user}

    # arrival times: mixture of a fast burst and a slow tail
    scale = tscale * np.exp(rng.normal(0, 0.45))
    burst = rng.random(n_rx) < 0.65
    times = np.where(burst, rng.exponential(scale * 0.5, n_rx), rng.exponential(scale * 6, n_rx))
    times = np.sort(np.clip(times, 0.05, 60 * 48))

    nodes = [(tid_base, 0.0, src_uid)]
    reactions, structure_parent = [], {}
    participants = [src_uid]
    for k, t in enumerate(times):
        rid = tid_base + k + 1
        if len(nodes) > 1 and rng.random() < p_deep:
            # reply to a recent reply (creates depth); recency-biased choice
            idx = len(nodes) - 1 - min(int(rng.exponential(2.0)), len(nodes) - 2)
            parent = nodes[idx]
        else:
            parent = nodes[0]
        # some users come back (reciprocity / conversation)
        if len(participants) > 2 and rng.random() < 0.18:
            uid = int(rng.choice(participants))
        else:
            uid = int(rng.choice(user_pool))
            participants.append(uid)
        created = t0 + timedelta(minutes=float(t))
        reactions.append({
            "id": rid, "id_str": str(rid), "text": "@reply",
            "created_at": created.strftime(FMT),
            "in_reply_to_status_id": parent[0], "in_reply_to_status_id_str": str(parent[0]),
            "user": user_obj(rng, uid, fol_mu, new_rate, ver_src * 0.4, created),
        })
        structure_parent[rid] = parent[0]
        nodes.append((rid, t, uid))

    # nested structure.json
    children = {}
    for c, p in structure_parent.items():
        children.setdefault(p, []).append(c)

    def nest(n):
        kids = children.get(n, [])
        return {str(k): nest(k) for k in kids} if kids else []
    structure = {str(tid_base): nest(tid_base)}
    return source, reactions, structure


def write_thread(base, source, reactions, structure, annotation):
    tdir = base / str(source["id"])
    (tdir / "source-tweets").mkdir(parents=True, exist_ok=True)
    (tdir / "reactions").mkdir(exist_ok=True)
    (tdir / "source-tweets" / f"{source['id']}.json").write_text(json.dumps(source))
    for r in reactions:
        (tdir / "reactions" / f"{r['id']}.json").write_text(json.dumps(r))
    (tdir / "structure.json").write_text(json.dumps(structure))
    (tdir / "annotation.json").write_text(json.dumps(annotation))


def main(scale: float = 1.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    root = SAMPLE_DIR / "all-rnr-annotated-threads"
    if root.exists():
        shutil.rmtree(root)
    user_pool = np.arange(10_000_000, 10_060_000)
    tid = 500_000_000_000_000_000
    ann_map = {"true": {"misinformation": "0", "true": "1"},
               "false": {"misinformation": "1", "true": "0"},
               "unverified": {"misinformation": "0", "true": "0"}}
    total = 0
    for event, counts in EVENTS.items():
        ev_dir = root / f"{event}-all-rnr-threads"
        for cls, n in zip(("true", "false", "unverified", "nonrumour"), counts):
            n = int(round(n * scale))
            sub = ev_dir / ("non-rumours" if cls == "nonrumour" else "rumours")
            for _ in range(n):
                src, rx, st = simulate_thread(rng, cls, event, tid, user_pool)
                ann = {"is_rumour": "nonrumour" if cls == "nonrumour" else "rumour"}
                ann.update(ann_map.get(cls, {}))
                write_thread(sub, src, rx, st, ann)
                tid += 1000
                total += 1
        print(f"  {event:<18} done")
    print(f"Synthetic PHEME-format data written: {total} threads -> {root}")


if __name__ == "__main__":
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(scale=s)
