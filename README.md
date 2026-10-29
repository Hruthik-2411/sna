# Early Viral Misinformation Detection Using Social Network Analysis

23 CSE 356 Social Network Analysis - Palivela Sai Hruthik (AM.SC.U4AIE23141), Nallapaneni Vaibhav (AM.SC.U4AIE23135)

**Question:** how early, and how accurately, can a false rumour be flagged using *only the structure of how it
spreads* (no text), before it goes viral?

## 1. Setup (once)

```bash
# Python 3.10+ recommended
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the dataset (PHEME veracity, 9 events)

1. Download **PHEME dataset for Rumour Detection and Veracity Classification** from figshare
   (https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078,
   a `.tar.bz2` archive, usually named `PHEME_veracity.tar.bz2`).
2. Extract it (7-Zip on Windows, `tar -xjf PHEME_veracity.tar.bz2` on macOS/Linux).
   Kaggle also hosts mirrors of the same folder ("PHEME dataset"); any copy that contains
   `*-all-rnr-threads` folders works.
3. Move the folder **`all-rnr-annotated-threads`** into `data/raw/`:

```
sna-misinfo/data/raw/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours/...
```

The loader searches `data/raw/` recursively, so extra nesting is fine. You can also point anywhere:
`python run_all.py --data "D:/datasets/PHEME"`.

## 3. Run everything

```bash
python run_all.py
```

Stages (about 10-20 minutes on a laptop):

| Stage | What happens | Output |
|---|---|---|
| 1 load | parse 6,425 threads + replies | `outputs/processed/threads.csv`, `tweets.pkl` |
| 2 features | build cascade + user graphs, truncate at 5/10/30/60/180 min/full, extract 33 features | `outputs/processed/features.csv` |
| 3 models | leave-one-event-out CV, 3 models, ablations, text baseline, importance | `outputs/results/*.csv` |
| 4 charts | all figures | `outputs/figures/*.png` |
| 5 demo | precompute showcase cascades | `outputs/demo/demo_cascades.json` |
| 6 report | Word report with all numbers filled in | `outputs/report/final_report.docx` |

Useful flags: `--skip-features` (reuse features), `--only report` (one stage), `--sample` (synthetic data).

## 4. Run the demo

```bash
streamlit run app/demo_app.py
```

Opens in the browser: Live replay, Race (A vs B), Why flagged?, Results.
Backup GIFs (if anything fails on stage): `python scripts/make_demo_gifs.py` -> `outputs/demo/*.gif`.

## 5. Trying it without PHEME

```bash
python run_all.py --sample
```

Generates synthetic data in the exact PHEME format and runs the full pipeline. Numbers are **not**
research results - use it only to test the setup and rehearse the demo.

## Project layout

```
config.py                 paths, windows, tasks
run_all.py                one command for the whole pipeline
src/load_pheme.py         step 1  dataset loader
src/graphs.py             step 2  cascade tree + user interaction graph, truncation
src/features.py           step 3  33 structure-only features
src/modeling.py           step 4  LOEO CV, models, baselines, importance
src/plots.py              step 5  charts
src/showcase.py           step 6  demo data
src/demo_viz.py           drawing for demo + GIFs
src/report.py             step 7  final report (.docx)
app/demo_app.py           Streamlit demo
scripts/make_sample_data.py   synthetic PHEME-format data
scripts/make_demo_gifs.py     backup GIFs
```
