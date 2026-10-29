"""
Run the whole project end to end:

    data -> graphs & features -> models -> charts -> demo data -> final report

Usage
    python run_all.py                     # real PHEME in data/raw/
    python run_all.py --sample            # synthetic PHEME-format data (generated if missing)
    python run_all.py --data D:/PHEME     # any folder containing *-all-rnr-threads
    python run_all.py --skip-features     # reuse outputs/processed/features.csv
    python run_all.py --only report       # just one stage: load|features|models|charts|demo|report
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

import config as C
from src import plots
from src.features import extract_all
from src.load_pheme import load_pheme, summarise
from src.modeling import run_structure_models, run_text_baseline, time_to_flag
from src.report import build_report
from src.showcase import build_demo_data

warnings.filterwarnings("ignore")
STAGES = ["load", "features", "models", "charts", "demo", "report"]


def banner(s):
    print("\n" + "=" * 70 + f"\n {s}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use synthetic PHEME-format data")
    ap.add_argument("--data", type=str, default=None, help="folder containing the PHEME events")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--only", choices=STAGES, default=None)
    args = ap.parse_args()

    t0 = time.time()
    data_dir = Path(args.data) if args.data else (C.SAMPLE_DIR if args.sample else C.RAW_DIR)
    if args.sample and not any(C.SAMPLE_DIR.rglob("*-all-rnr-threads")):
        banner("Generating synthetic sample data")
        sys.path.insert(0, str(C.ROOT / "scripts"))
        import make_sample_data
        make_sample_data.main()

    stages = [args.only] if args.only else STAGES
    if args.skip_features and "features" in stages:
        stages.remove("features")
    P, R, F, D = C.PROCESSED_DIR, C.RESULTS_DIR, C.FIG_DIR, C.DEMO_DIR

    # ---------------- 1. load --------------------------------------------------
    if "load" in stages:
        banner(f"1/6  Loading PHEME from {data_dir}")
        threads, tweets = load_pheme(data_dir)
        threads.to_csv(P / "threads.csv", index=False)
        tweets.to_pickle(P / "tweets.pkl")
        summ = summarise(threads)
        summ.to_csv(R / "dataset_summary.csv")
        print(summ)
        (R / "run_info.json").write_text(json.dumps({
            "data_dir": str(data_dir), "is_sample": bool(args.sample),
            "n_threads": int(len(threads)), "n_tweets": int(len(tweets))}, indent=1))

    threads = pd.read_csv(P / "threads.csv", dtype={"thread_id": str})
    threads["source_text"] = threads["source_text"].fillna("")
    tweets = pd.read_pickle(P / "tweets.pkl")

    # ---------------- 2. features ---------------------------------------------
    if "features" in stages:
        banner(f"2/6  Building graphs + extracting features for windows {C.WINDOWS}")
        feats = extract_all(threads, tweets, C.WINDOWS, seed=C.RANDOM_STATE)
        feats.to_csv(P / "features.csv", index=False)
    feats = pd.read_csv(P / "features.csv", dtype={"thread_id": str, "window": str})

    # ---------------- 3. models -------------------------------------------------
    if "models" in stages:
        banner("3/6  Leave-one-event-out modelling")
        all_m, all_p, all_i, all_e, txt, mains = [], [], [], [], [], {}
        for task in C.TASKS:
            print(f"\n--- {task}: {C.TASKS[task]}")
            tb, _ = run_text_baseline(threads, task, seed=C.RANDOM_STATE)
            print(f"  text baseline AUC={tb['auc']:.3f}  F1={tb['macro_f1']:.3f}")
            txt.append(tb)
            m, p, i, e, main = run_structure_models(feats, task, C.WINDOWS, seed=C.RANDOM_STATE,
                                                    main_model=C.MAIN_MODEL)
            all_m.append(m), all_p.append(p), all_i.append(i), all_e.append(e)
            mains[task] = main
        pd.concat(all_m).to_csv(R / "metrics.csv", index=False)
        pd.concat(all_p).to_csv(R / "oof_predictions.csv", index=False)
        pd.concat(all_i).to_csv(R / "importance.csv", index=False)
        pd.concat(all_e).to_csv(R / "per_event.csv", index=False)
        pd.DataFrame(txt).to_csv(R / "text_baseline.csv", index=False)
        info = json.loads((R / "run_info.json").read_text())
        info["main_model"] = mains
        (R / "run_info.json").write_text(json.dumps(info, indent=1))
        preds = pd.concat(all_p)
        pa = preds[(preds.task == "A_false_vs_true") & (preds.model == mains["A_false_vs_true"])]
        # time to flag: both classes (catches for false rumours, false alarms for true ones)
        time_to_flag(pa, C.WINDOWS).to_csv(R / "time_to_flag.csv", index=False)

    metrics = pd.read_csv(R / "metrics.csv", dtype={"window": str})
    preds = pd.read_csv(R / "oof_predictions.csv", dtype={"thread_id": str, "window": str})
    imp = pd.read_csv(R / "importance.csv", dtype={"window": str})
    per_event = pd.read_csv(R / "per_event.csv", dtype={"window": str})
    text_base = pd.read_csv(R / "text_baseline.csv")
    MAIN = json.loads((R / "run_info.json").read_text())["main_model"]      # {task: model}

    # ---------------- 4. charts -------------------------------------------------
    if "charts" in stages:
        banner("4/6  Charts")
        plots.pipeline_diagram(F / "pipeline.png")
        plots.dataset_overview(threads, F / "fig1_dataset.png")
        plots.cascade_growth(tweets, threads, F / "fig2_cascade_growth.png")
        for task, tag in (("A_false_vs_true", "A"), ("B_rumour_vs_nonrumour", "B")):
            plots.performance_vs_time(metrics, text_base, task, C.WINDOWS, F / f"fig3_auc_vs_time_{tag}.png",
                                      main_model=MAIN[task])
            plots.performance_vs_time(metrics, text_base, task, C.WINDOWS, F / f"fig3_f1_vs_time_{tag}.png",
                                      metric="macro_f1", main_model=MAIN[task])
            plots.ablation(metrics, task, C.WINDOWS, F / f"fig4_ablation_{tag}.png", MAIN[task])
            plots.importance_heatmap(imp, task, C.WINDOWS, F / f"fig5_importance_{tag}.png")
            plots.feature_distributions(feats, imp, task, F / f"fig6_distributions_{tag}.png")
            plots.per_event_chart(per_event, task, F / f"fig8_per_event_{tag}.png")
        ttf = pd.read_csv(R / "time_to_flag.csv")
        plots.time_to_flag_chart(ttf, C.WINDOWS, F / "fig7_time_to_flag.png")
        print(f"  charts saved to {F}")

    # ---------------- 5. demo data -----------------------------------------------
    if "demo" in stages:
        banner("5/6  Precomputing demo cascades")
        cas = build_demo_data(feats, tweets, threads, preds, imp, D / "demo_cascades.json",
                              main_model=MAIN["A_false_vs_true"])
        print(f"  {len(cas)} showcase cascades -> {D / 'demo_cascades.json'}")

    # ---------------- 6. report -------------------------------------------------
    if "report" in stages:
        banner("6/6  Final report")
        info = json.loads((R / "run_info.json").read_text())
        if "runtime_min" not in info or args.only is None:
            info["runtime_min"] = round((time.time() - t0) / 60, 1)
            (R / "run_info.json").write_text(json.dumps(info, indent=1))
        out = build_report({"results": R, "figures": F}, C.WINDOWS,
                           C.REPORT_DIR / "final_report.docx", is_sample=info.get("is_sample", False),
                           main=MAIN)
        print(f"  report -> {out}")

    # ---------------- headline ----------------------------------------------------
    a = metrics[(metrics.task == "A_false_vs_true") & (metrics.model == MAIN["A_false_vs_true"]) &
                (metrics.feature_set == "all")].set_index("window")["auc"]
    banner(f"Headline (Task A: false vs true, AUC, main model {MAIN['A_false_vs_true']})")
    print("  " + "   ".join(f"{w}: {a.get(str(w), float('nan')):.3f}" for w in C.WINDOWS))
    print(f"  text baseline: {text_base.set_index('task').at['A_false_vs_true', 'auc']:.3f}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  Next: streamlit run app/demo_app.py")


if __name__ == "__main__":
    main()
