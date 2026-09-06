This folder contains the actual output of a full, real end-to-end pipeline
run (see full_pipeline_run.log for the complete console output, including
all measured precision/recall/AUC numbers referenced in TECHNICAL_WRITEUP.md).

These files are provided as evidence the system works, so you can inspect
real results immediately without needing GeoLite2 set up first. To
regenerate everything yourself from scratch, run `python3 run_pipeline.py`
from the repo root — it will overwrite these files with a fresh run.

Key file to look at first: final_alerts_explained.csv — the final ranked,
SHAP-explained risk score per wallet, exactly as the dashboard (app.py) reads it.
