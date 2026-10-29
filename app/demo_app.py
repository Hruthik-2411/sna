"""
Live demo:  streamlit run app/demo_app.py

Pages
  1. Live replay    - a held-out cascade grows in (accelerated) real time; risk meter updates
  2. Race           - a false and a true story spread side by side; audience guesses
  3. Why flagged?   - which network signals pushed the risk up or down
  4. Results        - the final charts and numbers
Everything is precomputed by run_all.py, so the demo works offline.
"""
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEMO_DIR, FIG_DIR, RESULTS_DIR  # noqa: E402
from src.demo_viz import (WINDOW_ORDER, frame, race_frame, radial_layout,  # noqa: E402
                          replay_times, why_chart)

INF = float("inf")
st.set_page_config(page_title="Early Misinformation Detection - SNA", layout="wide")


@st.cache_data
def load_demo():
    p = DEMO_DIR / "demo_cascades.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data
def load_results():
    out = {}
    for name in ("metrics", "text_baseline", "per_event"):
        f = RESULTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(f, dtype={"window": str}) if f.exists() else None
    info = RESULTS_DIR / "run_info.json"
    out["info"] = json.loads(info.read_text()) if info.exists() else {}
    return out


demo = load_demo()
res = load_results()
if demo is None:
    st.error("No demo data found. Run `python run_all.py` (or `python run_all.py --sample`) first.")
    st.stop()

cascades = demo["cascades"]
for i, c in enumerate(cascades):
    c["case"] = f"Case {i + 1:02d} - {c['event']} - {c['n_reactions']} replies"
by_case = {c["case"]: c for c in cascades}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Early Misinformation\nDetection with SNA")
page = st.sidebar.radio("Demo", ["1 - Live replay", "2 - Race: A vs B", "3 - Why flagged?",
                                 "4 - Results", "About the project"])
speed = st.sidebar.slider("Replay speed (seconds per frame)", 0.03, 0.4, 0.12, 0.01)
n_frames = st.sidebar.slider("Frames per replay", 30, 120, 60, 10)
if res["info"].get("is_sample"):
    st.sidebar.warning("Running on SYNTHETIC sample data. Re-run the pipeline on PHEME for real results.")
MAIN = res["info"].get("main_model", {}).get("A_false_vs_true", demo.get("model", "RandomForest"))
st.sidebar.caption(f"Model: structure-only {MAIN}, trained with leave-one-event-out: "
                   "every cascade shown here comes from an event the model never saw.")


def max_time(c):
    return min(max(n["t"] for n in c["nodes"]), 180.0)


def legend_note():
    st.caption("Black = source tweet | blue = reply | orange = new in the last 3 minutes | "
               "yellow ring = verified account | dot size = follower count")


# ---------------------------------------------------------------- 1. live replay
if page.startswith("1"):
    st.header("Live replay: can we flag it before it goes viral?")
    st.write("We replay a real conversation minute by minute. The model sees **only the network** - "
             "who replies, how fast, how deep - never the words. Its risk estimate updates at "
             "5, 10, 30, 60 and 180 minutes.")
    choice = st.selectbox("Pick a held-out cascade (labels hidden)", list(by_case))
    c = by_case[choice]
    pos = radial_layout(c["nodes"])
    col1, col2 = st.columns([1, 1])
    play = col1.button("Play replay", type="primary")
    manual = col2.toggle("Scrub manually instead")
    ph = st.empty()
    legend_note()

    if manual:
        t = st.slider("Minutes since source tweet", 0.0, float(max_time(c)), 10.0, 0.5)
        fig, _ = frame(c, t, pos)
        ph.pyplot(fig)
        plt.close(fig)
        if st.button("Reveal verdict"):
            fig, _ = frame(c, INF, pos, reveal=True)
            ph.pyplot(fig)
            plt.close(fig)
            st.info(f"Source tweet (the model never read this): \"{c['text']}\"")
    elif play:
        for t in replay_times(max_time(c), n_frames):
            fig, _ = frame(c, t, pos)
            ph.pyplot(fig)
            plt.close(fig)
            time.sleep(speed)
        time.sleep(0.8)
        fig, _ = frame(c, INF, pos, reveal=True)
        ph.pyplot(fig)
        plt.close(fig)
        p10 = c["probs"].get("10")
        verdict = c["label"].upper()
        if c["label"] == "false" and p10 is not None and p10 >= 0.5:
            st.success(f"Fact-check verdict: {verdict}. The network alone flagged it at minute 10 "
                       f"(risk {p10:.0%}).")
        elif c["label"] == "true" and p10 is not None and p10 < 0.5:
            st.success(f"Fact-check verdict: {verdict}. The model correctly kept it below the flag "
                       f"threshold (risk {p10:.0%} at 10 min).")
        else:
            st.warning(f"Fact-check verdict: {verdict}. The model got this one wrong - a good "
                       "example of the limits of structure-only detection.")
        st.info(f"Source tweet (the model never read this): \"{c['text']}\"")
    else:
        fig, _ = frame(c, 0.0, pos)
        ph.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------- 2. race
elif page.startswith("2"):
    st.header("Race: one is false, one is true. Can you tell which?")
    st.write("Two stories spread side by side on the same clock. Ask the audience to guess, "
             "then watch what the network-only model says.")
    fals = [c for c in cascades if c["kind"] == "false"]
    tru = [c for c in cascades if c["kind"] == "true"]
    if "pair" not in st.session_state or st.button("New pair"):
        f, t = random.choice(fals), random.choice(tru)
        st.session_state.pair = [f, t] if random.random() < 0.5 else [t, f]
    a, b = st.session_state.pair
    pa, pb = radial_layout(a["nodes"]), radial_layout(b["nodes"])
    ph = st.empty()
    if st.button("Start the race", type="primary"):
        tmax = max(max_time(a), max_time(b))
        for t in replay_times(tmax, n_frames):
            fig = race_frame(a, b, t, pa, pb)
            ph.pyplot(fig)
            plt.close(fig)
            time.sleep(speed)
        fig = race_frame(a, b, INF, pa, pb, reveal=True)
        ph.pyplot(fig)
        plt.close(fig)
        st.write(f"**Story A** ({a['event']}): \"{a['text']}\"  \n"
                 f"**Story B** ({b['event']}): \"{b['text']}\"")
    else:
        fig = race_frame(a, b, 0.0, pa, pb)
        ph.pyplot(fig)
        plt.close(fig)
    legend_note()

# ---------------------------------------------------------------- 3. why
elif page.startswith("3"):
    st.header("Why did the model flag it?")
    st.write("For the most important features at the chosen time, we place this cascade between a "
             "typical TRUE and a typical FALSE cascade (class medians).")
    choice = st.selectbox("Cascade", list(by_case))
    c = by_case[choice]
    w = st.select_slider("Observation window", options=[x for x in WINDOW_ORDER if x in c["explain"]],
                         value="30" if "30" in c["explain"] else WINDOW_ORDER[-1])
    p = c["probs"].get(w)
    left, right = st.columns([2, 1])
    rows = c["explain"].get(w, [])
    with left:
        if rows:
            fig = why_chart(rows, f"Top signals at {w}{'' if w == 'full' else ' min'}")
            st.pyplot(fig)
            plt.close(fig)
    with right:
        if p is not None:
            st.metric("Misinformation risk", f"{p:.0%}", "FLAG" if p >= 0.5 else "not flagged",
                      delta_color="inverse" if p >= 0.5 else "normal")
        n_false = sum(r["leans"] == "false" for r in rows)
        st.write(f"**{n_false} of {len(rows)}** top signals look more like a false rumour.")
        for r in rows:
            arrow = "towards FALSE" if r["leans"] == "false" else "towards TRUE"
            st.write(f"- {r['label']}: **{r['value']:.2f}** (true ~ {r['true_median']:.2f}, "
                     f"false ~ {r['false_median']:.2f}) -> {arrow}")
        if st.toggle("Reveal ground truth"):
            st.info(f"Verdict: **{c['label'].upper()}**  \nSource tweet: \"{c['text']}\"")

# ---------------------------------------------------------------- 4. results
elif page.startswith("4"):
    st.header("Results")
    m, tb = res["metrics"], res["text_baseline"]
    if m is not None:
        a = m[(m.task == "A_false_vs_true") & (m.model == MAIN) &
              (m.feature_set == "all")].set_index("window")
        cols = st.columns(5)
        for col, w in zip(cols, ["10", "30", "60", "full"]):
            col.metric(f"AUC @ {w}{'' if w == 'full' else ' min'}", f"{a.at[w, 'auc']:.3f}")
        cols[4].metric("Text baseline AUC", f"{tb.set_index('task').at['A_false_vs_true', 'auc']:.3f}")
    tabs = st.tabs(["Performance vs time", "Ablation", "Feature importance", "Time to flag",
                    "Per event", "Data", "Task B"])
    figs = [["fig3_auc_vs_time_A.png", "fig3_f1_vs_time_A.png"], ["fig4_ablation_A.png"],
            ["fig5_importance_A.png", "fig6_distributions_A.png"], ["fig7_time_to_flag.png"],
            ["fig8_per_event_A.png"], ["fig1_dataset.png", "fig2_cascade_growth.png"],
            ["fig3_auc_vs_time_B.png", "fig4_ablation_B.png", "fig5_importance_B.png"]]
    for tab, fl in zip(tabs, figs):
        with tab:
            for f in fl:
                if (FIG_DIR / f).exists():
                    st.image(str(FIG_DIR / f))
    if m is not None:
        with st.expander("Full metrics table"):
            st.dataframe(m.round(3))

# ---------------------------------------------------------------- about
else:
    st.header("Early Viral Misinformation Detection Using Social Network Analysis")
    st.markdown("""
**Problem.** False rumours spread faster than corrections. Most detectors read the *text* (easy to fake)
or wait for the *full cascade* (too late).

**Our question.** How early, and how accurately, can misinformation be flagged using **only the structure
of how it spreads**?

**Novelty.**
1. Early-window analysis (from virality prediction) applied to *veracity*.
2. Structure-only by design - the contribution of the network is isolated.
3. Performance reported as a curve over time, not one number on finished cascades.
4. Leave-one-event-out evaluation - tested on breaking news the model has never seen.

**Data.** PHEME veracity dataset - 9 breaking-news events, 6,425 Twitter conversation threads.
""")
    if (FIG_DIR / "pipeline.png").exists():
        st.image(str(FIG_DIR / "pipeline.png"))
