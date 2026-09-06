"""
run_pipeline.py

Day 14 build — runs the ENTIRE pipeline end-to-end as one command, stage by
stage, with timing and clear pass/fail reporting. This is the actual proof
that the system is "a workable complete offline solution" rather than a
collection of scripts that happened to work when run manually in the right
order across many separate sessions.

Usage:  python3 run_pipeline.py
Stops immediately on the first stage that fails, with the real error visible
(not swallowed), so a broken stage is always obvious.
"""

import subprocess
import sys
import time

STAGES = [
    ("Day 3  - Dataset generation",        "generate_dataset.py"),
    ("Day 4  - Graph construction",        "graph_builder.py"),
    ("Day 5  - CIOH + change-address clustering", "change_address_heuristic.py"),
    ("Day 6  - Wallet feature engineering", "wallet_features.py"),
    ("Day 7  - Isolation Forest anomaly detection", "anomaly_detection.py"),
    ("Day 8  - Peeling/CoinJoin graph-traversal detection", "peeling_chain_detection.py"),
    ("Day 9  - Peeling-chain classifier", "peeling_chain_classifier.py"),
    ("Day 10 - Risk propagation + combined scoring", "risk_propagation.py"),
    ("Day 11 - SHAP explainability", "explainability.py"),
    ("Day 11 - Final explained alert list", "final_alerts.py"),
]


def run_stage(name: str, script: str) -> tuple:
    print(f"\n{'='*70}\n{name}  ({script})\n{'='*70}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    elapsed = time.time() - start

    print(result.stdout[-2000:])  # last 2000 chars is enough to see the key metrics per stage
    if result.returncode != 0:
        print(f"\n!!! STAGE FAILED ({elapsed:.1f}s) !!!")
        print(result.stderr[-3000:])
        return False, elapsed
    print(f"--- Stage completed in {elapsed:.1f}s ---")
    return True, elapsed


if __name__ == "__main__":
    print("Starting full pipeline run...")
    total_start = time.time()
    results = []

    for name, script in STAGES:
        ok, elapsed = run_stage(name, script)
        results.append((name, ok, elapsed))
        if not ok:
            print(f"\nPipeline stopped: '{name}' failed. Fix the error above before continuing.")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}\nPIPELINE COMPLETE -- all {len(STAGES)} stages passed in {total_elapsed:.1f}s\n{'='*70}")
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}  ({elapsed:.1f}s)")

    print("\nDashboard: run 'streamlit run app.py' separately (needs a browser/tunnel, not part of this batch run).")
