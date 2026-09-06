# ChainSentinel

### AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Offline prototype for SIH Problem Statement 5 (National Technical Research
Organisation) — ingests Bitcoin-shaped transaction + network metadata,
builds a wallet/transaction/IP graph, applies four independent detection
methods (entity clustering, anomaly detection, peeling-chain/mixing
detection, risk propagation), and combines them into a single ranked,
explainable, evidence-backed risk score per wallet — viewable in an
interactive dashboard.

See `TECHNICAL_WRITEUP.md` for approach, model choices, evaluation results,
and known limitations.

## Requirements

- Python 3.10+
- Linux (tested offline, no external API calls required at runtime)
- Free MaxMind GeoLite2 account (for real GeoIP resolution — optional; a
  deterministic synthetic fallback is used automatically if absent)

## Setup

```bash
pip install pandas numpy networkx scikit-learn shap streamlit streamlit-agraph geoip2 --break-system-packages
```

### GeoLite2 (optional but recommended)
1. Sign up free: https://www.maxmind.com/en/geolite2/signup
2. Under "Manage License Keys", generate a key, then download:
   `GeoLite2-City.mmdb` and `GeoLite2-ASN.mmdb`
3. Place both in `./geoip_data/`

**Do not commit the `.mmdb` files to this repo** — MaxMind's license
prohibits redistribution. `geoip_data/` is already in `.gitignore`.
If the files aren't present, the pipeline still runs fully offline using a
deterministic synthetic fallback (clearly logged as such).

## Running the full pipeline

```bash
python3 run_pipeline.py
```

Regenerates the dataset, graph, all models, and the final explained alert
list from scratch in under a minute. Stops immediately with the real error
visible if any stage fails.

### Running stages individually
If you want to inspect intermediate output at each step instead of the
batch run, run in this exact order (each stage reads files written by the
previous one):

```bash
python3 generate_dataset.py        # -> btc_dataset.json / .csv
python3 graph_builder.py           # -> btc_graph.gpickle / btc_graph_wallet_only.gpickle
python3 change_address_heuristic.py # -> predicted_entity_id.pkl / _v2.pkl
python3 wallet_features.py         # -> wallet_features.csv
python3 anomaly_detection.py       # -> wallet_anomaly_scores.csv
python3 peeling_chain_detection.py # -> detected_peeling_chains.pkl
python3 peeling_chain_classifier.py # -> peeling_classifier.pkl
python3 risk_propagation.py        # -> wallet_final_risk_scores.csv
python3 explainability.py          # -> wallet_shap_explanations.csv / chain_shap_explanations.csv
python3 final_alerts.py            # -> final_alerts_explained.csv  (dashboard reads this)
```

## Running the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `final_alerts_explained.csv`,
`btc_graph_wallet_only.gpickle`, and `graph_view.py` to be present (produced
by the pipeline above).

## Repository structure

| File | Purpose |
|---|---|
| `data_schema.md` | Field-level schema and entity-relationship model |
| `geoip_lookup.py` | Real GeoLite2 wrapper with synthetic offline fallback |
| `generate_dataset.py` | Synthetic Bitcoin-shaped dataset generator (all patterns + ground truth) |
| `graph_builder.py` | Builds the wallet/tx/IP NetworkX graph |
| `union_find.py` | Disjoint-set data structure |
| `cioh_clustering.py` | Common-Input-Ownership Heuristic entity clustering |
| `change_address_heuristic.py` | Change-address heuristic, extends CIOH clustering |
| `wallet_features.py` | Per-wallet feature engineering |
| `anomaly_detection.py` | Isolation Forest training + evaluation |
| `peeling_chain_detection.py` | Graph-traversal peeling-chain and CoinJoin detectors |
| `peeling_chain_classifier.py` | Random Forest peeling-likelihood classifier |
| `risk_propagation.py` | Personalized PageRank + combined noisy-OR risk score |
| `explainability.py` | SHAP explanations for both ML models |
| `final_alerts.py` | Merges SHAP reasons into the final ranked alert list |
| `graph_view.py` | Focused subgraph builder for the dashboard's graph view |
| `app.py` | Streamlit dashboard |
| `run_pipeline.py` | End-to-end orchestrator |
| `TECHNICAL_WRITEUP.md` | Full approach, model choices, results, limitations |

## Known limitations

See `TECHNICAL_WRITEUP.md` Section 5 for a full, honest account — including
small positive-class sample sizes, the change-address heuristic's precision
trade-off, and why CoinJoin detection is currently rule-based rather than ML-based.
