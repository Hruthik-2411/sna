"""
Central configuration for the Early Viral Misinformation Detection project.
Edit paths / windows here; every script imports from this file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# Put the extracted PHEME folder (all-rnr-annotated-threads) inside data/raw/.
# The loader searches RAW_DIR recursively for folders ending in "-all-rnr-threads",
# so the exact nesting does not matter.
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"          # generated test data (same format)

OUT_DIR = ROOT / "outputs"
PROCESSED_DIR = OUT_DIR / "processed"          # tweets.csv, threads.csv, features.csv
RESULTS_DIR = OUT_DIR / "results"              # metrics, predictions, importances
FIG_DIR = OUT_DIR / "figures"                  # all charts (PNG)
DEMO_DIR = OUT_DIR / "demo"                    # precomputed demo cascades + GIFs
REPORT_DIR = OUT_DIR / "report"                # final_report.docx

# ---------------------------------------------------------------------------
# Early-detection windows (minutes after the source tweet).  "full" = whole cascade.
# The proposal's windows are 10 / 30 / 60 / full; 5 and 180 add resolution to the curve.
# ---------------------------------------------------------------------------
WINDOWS = [5, 10, 30, 60, 180, "full"]

# ---------------------------------------------------------------------------
# Classification tasks
# ---------------------------------------------------------------------------
# A (main):       false vs true rumours       (unverified rumours dropped)
# B (supporting): rumour vs non-rumour         (all 9 events, larger sample)
TASKS = {
    "A_false_vs_true": "Misinformation (false) vs true rumours",
    "B_rumour_vs_nonrumour": "Rumour vs non-rumour",
}

NEW_ACCOUNT_DAYS = 90        # an account younger than this counts as "new"
RANDOM_STATE = 42

# Main model: "auto" = the classifier with the best mean AUC across windows (per task).
# Or force one of: "LogReg", "RandomForest", "GradBoost".
MAIN_MODEL = "auto"

for d in (PROCESSED_DIR, RESULTS_DIR, FIG_DIR, DEMO_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
