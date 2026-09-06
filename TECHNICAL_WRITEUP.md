# ChainSentinel
## AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic — Technical Write-up

**Category:** Software | **Theme:** Cryptocurrency | **Organisation:** National Technical Research Organisation

---

## 1. Approach

This prototype ingests bulk Bitcoin-shaped transaction metadata (network-layer:
IP/port/timing; blockchain-layer: wallet/TXID/amount), builds a unified graph
linking wallets, transactions, and peer IPs, and applies four independent
detection methods whose outputs are combined into a single ranked,
explainable risk score per wallet. The full pipeline runs offline on Linux
with no external API calls.

**Pipeline stages:**

```
Synthetic Data Generator (Bitcoin-shaped, ground-truth-labeled)
        |
Ingestion & Graph Construction (NetworkX MultiDiGraph: wallet/tx/IP nodes)
        |
   +----+----+----+
   |    |    |    |
 CIOH  Iso-  Peel- Risk
 +chg  lation ing  Propa-
 -addr Forest RF   gation (PageRank, seeded from known cash-out wallets)
   |    |    |    |
   +----+----+----+
        |
  Combined Risk Score (noisy-OR) + SHAP Explanations
        |
  Streamlit Dashboard (ranked table + link-analysis graph view)
```

### 1.1 Why synthetic, ground-truth data
No real seized/live-intercept dataset was provided (per problem statement).
A generator was built that produces **Bitcoin-shaped synthetic data** (real
UTXO ledger semantics, real GeoLite2 IP resolution, real entity/address
relationships) with every illicit pattern's ground truth embedded at
generation time. This enables actual precision/recall measurement throughout
-- every number in this document is measured against known ground truth, not
asserted.

**Final dataset**: 1,007 transactions, 965 wallets, 65 real-world entities
(60 normal + 5 reserved cash-out/mule sinks), 50 peer IPs.
- 856 normal transactions
- 113 peeling-chain transactions (18 chains, 3 timing styles: tight/medium/loose)
- 29 fan-in + fan-out transactions (layering)
- 5 CoinJoin-like mixing transactions
- 56 **hard-negative** pass-through transactions (14 chains) -- legitimate
  multi-hop payment forwarding, structurally similar to peeling but with
  genuine ownership change each hop, deliberately added to stress-test the
  peeling classifier's generalization rather than let it learn a trivial
  shortcut

### 1.2 Honest data-generation caveats
- IP fields (`src_ip`/`dst_ip`) represent an **observed P2P relay event**,
  not proven sender/receiver identity -- a Bitcoin transaction contains no
  IP field; this is standard practice in real deanonymization research
  (e.g. peer-observation attacks), and is treated as a probabilistic
  correlation throughout, never an ownership claim.
- GeoIP resolution uses real MaxMind GeoLite2 databases: **88.0% of unique
  peer IPs resolved to real geographic data**, 12.0% fell back to a
  deterministic synthetic mapping (legitimately unallocated IP ranges --
  comparable to real-world GeoIP coverage gaps).
- Peer IP selection uses a **peer-affinity model**: each entity has a
  consistent home pool of 2-3 geographically-clustered peers, with designed
  exceptions (14.9% of transactions) concentrated in peeling/CoinJoin flows
  -- this makes geo-velocity a genuine, causally-explainable signal rather
  than random noise (confirmed: 62.9% geo-exception rate in peeling
  transactions vs. 6.75% in normal ones).

---

## 2. Model Choices & Rationale

### 2.1 Entity Clustering -- Common-Input-Ownership Heuristic (CIOH) + Change-Address Heuristic
**Method:** Union-Find over co-spent transaction inputs (CIOH), extended with
a causally-computed change-address heuristic (address freshness signal,
confident only when exactly one of two outputs is fresh).

**Why:** CIOH is the foundational heuristic in real blockchain forensics.
Measuring its actual limitations (rather than assuming it works) is more
scientifically honest and more informative for judges than presenting it as
a solved problem.

**Results (measured against ground truth):**

| | Precision | Recall | F1 |
|---|---|---|---|
| CIOH only | 0.519 | 0.060 | 0.108 |
| CIOH + change-address | 0.288 | 0.284 | 0.286 |

Recall improved 3.7x by linking addresses across time (not just within one
transaction), at a real precision cost. **Root cause of the precision drop,
found through this project's own testing:** the freshness-based heuristic
occasionally misfires on legitimate multi-recipient payments that happen to
have one fresh and one reused address -- an honest, documented limitation of
using a single weak signal rather than multiple corroborating ones (a known
improvement path: combine freshness with address-type/script consistency
and amount-pattern signals).

CoinJoin transactions defeat CIOH **100% of the time** by design (44/44
merge pairs were false), quantitatively justifying the need for a dedicated
statistical mixing detector (Section 2.3).

### 2.2 Anomaly Detection -- Isolation Forest
**Method:** Unsupervised Isolation Forest over 9 per-wallet features (amount
z-scores, transaction frequency, fan-in/out degree, geo-velocity,
time-of-day entropy), with signed-log transformation of skewed features
before training.

**Why Isolation Forest:** No assumption about feature distribution shape is
needed (unlike z-score-threshold methods); isolates anomalies by how few
random splits are needed to separate them, which suits a mix of skewed
count/frequency/velocity features well.

**Results:** AUC-PR 0.636 (vs. 0.287 base rate), AUC-ROC 0.793.

**Per-pattern breakdown (top 20% reviewed):** peeling 59%, fan-out 50%,
fan-in 39%, **CoinJoin only 16%**. This is an honest, expected finding:
CoinJoin's defining signature (multiple equal-value outputs *within one
transaction*, across independent entities) isn't visible in any per-wallet
aggregate feature -- confirming the architectural need for a dedicated
mixing detector rather than relying on anomaly detection alone.

### 2.3 Peeling-Chain & Mixing Detection -- Graph Traversal + Random Forest
**Method (Day 8, rule-based):**
- *CoinJoin*: flag transactions with >=3 near-equal-value outputs (coefficient
  of variation < 0.05) and >=2 distinct inputs. **Result: 100% precision,
  100% recall** -- the statistical signature is unambiguous.
- *Peeling chains*: graph traversal following asymmetric 2-output
  transactions (small "peel" + large "remainder") forward through time,
  as long as the remainder keeps getting re-spent asymmetrically.
  **Result: 100% recall, 52.8% precision** (18/18 true chains found, 17
  coincidental normal-traffic look-alikes also flagged).

**Method (Day 9, ML on top of traversal):** the 52.8% precision above proves
graph traversal alone cannot distinguish real peeling from coincidence --
motivating a Random Forest classifier on chain-level features (timing gaps,
peel-ratio consistency, hop count, fee pattern).

**Results:**

| Evaluation | Precision | Recall |
|---|---|---|
| Leave-One-Out CV (36 samples) | 0.950 | 1.000 |
| **Held-out test set** (25 train / 11 test, never seen during training) | **0.750** | **1.000** |

The held-out result is reported as the primary number -- it is a genuine
generalization test, deliberately including **hard-negative** pass-through
chains (legitimate multi-hop forwarding that structurally resembles peeling)
to avoid the classifier learning a trivial "hop-count" shortcut. SHAP
analysis confirms the model learned the correct causal signal: timing-gap
features (`min_time_gap_sec`, `mean_time_gap_sec`, `max_time_gap_sec`)
account for 70% of feature importance -- consistent hop timing is what
separates automated peeling from irregular legitimate forwarding.

### 2.4 Risk Propagation -- Personalized PageRank
**Method:** Personalized PageRank over the wallet+transaction graph
(IP layer excluded -- it would let risk leak between unrelated wallets
sharing a relay peer), seeded from a **small, realistic** set of 6 known
cash-out wallets (simulating external threat intel, not full ground truth),
weighted by transaction amount, with a lowered damping factor (alpha=0.55) for
faster hop-decay than general web PageRank.

### 2.5 Final Combined Risk Score -- Noisy-OR
```
final_risk = 1 - (1 - risk_propagated) x (1 - risk_anomaly) x (1 - risk_peeling)
```
Chosen over a plain average because any single strong signal should raise
suspicion -- averaging would let two weak/irrelevant signals dilute one
genuinely strong one.

**Result: combining outperforms every individual signal.**

| Signal | AUC-PR |
|---|---|
| Propagated risk alone | 0.362 |
| Anomaly score alone | 0.636 |
| Peeling score alone | 0.782 |
| **Combined (noisy-OR)** | **0.868** |

Ties at the ceiling (multiple wallets at risk=1.000) are broken by a
secondary "total evidence" score (sum of the three raw signals), so a
wallet flagged by all three methods ranks above one flagged by only one.

---

## 3. Explainability Method

Every alert carries a **SHAP-based, per-feature reason string**, not a bare
score:
- **Isolation Forest**: `shap.TreeExplainer` (works natively on tree-based
  anomaly models) attributes each wallet's anomaly score to specific input
  features.
- **Peeling-chain Random Forest**: `shap.TreeExplainer` on the classifier's
  P(peeling) output, attributing per-chain predictions to specific timing/
  ratio/fee features.
- **Propagated risk**: not a supervised model, so no SHAP needed -- its
  explanation is inherently transparent (the actual graph path to a known
  cash-out wallet).

Example real output from this system:
> `[Anomaly] Driven by: an unusually high transaction frequency (contributed +0.089). [Peeling-chain, P=1.00] Driven by: how tightly-timed the fastest hop was (contributed +0.173); the average time between hops (contributed +0.122).`

This lets an investigator see *why* a wallet was flagged, not just *that* it
was -- directly satisfying the "explainable, evidence-backed alert" deliverable.

---

## 4. Dashboard

A Streamlit dashboard (`app.py`) provides:
- Summary metrics (wallets analyzed, flagged counts, average risk)
- Filterable, sortable ranked alert table with full SHAP explanations
- An interactive link-analysis graph view (`streamlit-agraph`), colored by
  risk score, with click-to-drill-down per wallet/transaction -- deliberately
  scoped to a focused neighborhood (top-K riskiest wallets or a searched
  wallet's 1-hop context) rather than the full ~2,000-node graph, which
  would be an unreadable blob.

---

## 5. Known Limitations (stated plainly, not hidden)

1. **Small positive-class sample sizes.** 18 peeling chains and 5 CoinJoin
   transactions is enough to demonstrate the method but not enough for
   fully robust statistical confidence -- Leave-One-Out and held-out splits
   are used specifically because a large train/test split isn't meaningful
   here; more synthetic volume (or real seized data) would strengthen this.
2. **Change-address heuristic precision gap** (Section 2.1) -- a single
   freshness signal is too weak alone; combining with amount-pattern and
   address-type signals is the natural next step.
3. **IP-to-wallet correlation is probabilistic, not identity.** Explicitly
   modeled and documented throughout -- no claim in this system asserts an
   IP address proves who controls a wallet.
4. **Address-level (not full UTXO-level) traversal** in peeling-chain
   detection can occasionally misattribute continuation when an address is
   reused -- a known simplification versus full UTXO-graph forensic tools.
5. **CoinJoin detection is currently rule-based only** (no ML layer) --
   justified because the statistical signature is unambiguous (100%/100%
   measured), but a rule-based detector is easier to evade with deliberate
   amount jitter than a learned one would be.

---

## 6. Reproducibility

Full pipeline (`run_pipeline.py`) regenerates the dataset, graph, all
models, and final alerts from scratch in under 40 seconds on a standard
Linux machine, entirely offline (once GeoLite2 `.mmdb` files are placed
locally per MaxMind's license -- not redistributed in this repo). See
`README.md` for exact setup and run instructions.
