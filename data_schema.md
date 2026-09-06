# Data Schema — AI-Powered Bitcoin Transaction Monitoring

## 1. Raw Input Fields (as required by the problem statement)

| Field | Type | Description | Bitcoin Concept it Maps To |
|---|---|---|---|
| `timestamp` | datetime | When the transaction/network event was observed | Block time / relay time |
| `src_ip` | string (IPv4/IPv6) | IP of the peer **observed** sending/relaying the tx (not necessarily the originator) | Network layer — a relay observation, NOT wallet ownership |
| `dst_ip` | string (IPv4/IPv6) | IP of the peer **observed** receiving/relaying the tx | Network layer — a relay observation, NOT wallet ownership |
| `src_port` | int | Source port | Network layer metadata |
| `dst_port` | int | Destination port | Network layer metadata |
| `txid` | string (hash) | Unique transaction ID | Blockchain layer — the transaction itself |
| `input_addresses[]` | list of strings | Wallet addresses whose UTXOs are being spent | Inputs — used for CIOH clustering |
| `output_addresses[]` | list of strings | Wallet addresses receiving funds | Outputs — recipients + change address |
| `input_amounts[]` | list of floats (BTC) | Amount contributed by each input | Used for peeling-chain / amount-flow tracking |
| `output_amounts[]` | list of floats (BTC) | Amount sent to each output | Used for peeling-chain / CoinJoin detection |
| `geo_country` | string (ISO code) | Country resolved from IP via GeoIP DB | Network layer — geo-velocity anomaly detection |
| `asn` | string | Autonomous System Number resolved from IP | Network layer — proxy/VPN/hosting detection |

## 2. Prevout / UTXO Reference Fields (generator-level ground truth)

Note: all data below is **Bitcoin-SHAPED synthetic data** — fake addresses/txids/IPs of
the correct shape, not real chain data or real cryptography.

| Field | Type | Description |
|---|---|---|
| `input_prevouts[]` | list of `"txid:vout"` strings | The actual previous output each input spends — this is how a real Bitcoin parser links transactions, not by address alone |
| `sender_entity_ground_truth` | int | The true real-world entity that authored this transaction (known because we generated it) |
| `output_entity_ground_truth[]` | list of int, parallel to `output_addresses[]` | The true entity that owns each output |
| `is_change_ground_truth[]` | list of bool, parallel to `output_addresses[]` | Whether each output is change returning to the sender's own entity |
| `coinjoin_ground_truth` | bool | Whether this transaction is an injected CoinJoin-like mix (always `False` in the Day 2 normal-only generator; Day 3 sets `True` for injected ones) |

**Important:** these are generation-time facts, not model output. Keep them in a
separate column from anything your CIOH heuristic, clustering model, or classifier
later predicts (e.g. a predicted `is_change_output` or `predicted_entity_id`) — mixing
ground truth and predictions in the same field makes it impossible to measure accuracy.

## 3. Derived / Internal Fields (your models compute these — predictions, not ground truth)

| Field | Computed From | Used For |
|---|---|---|
| `fee` | sum(input_amounts) − sum(output_amounts) | Sanity check + feature for anomaly model (emitted by generator directly) |
| `predicted_entity_id` | CIOH clustering (Union-Find over co-spent inputs) | Your model's guess at grouping wallets — compare against `*_entity_ground_truth` |
| `predicted_is_change` | Heuristic/model: smallest/unlabeled output often = change | Compare against `is_change_ground_truth` |
| `chain_id` | Graph traversal (peeling-chain detection) | Tags transactions belonging to a suspected peeling sequence |

## 3. How the Fields Relate to Each Other (Entity Relationship Model)

```
WALLET (address)
  │
  ├──[spends from]──> TRANSACTION (txid)  [as input_address]
  │
  └──[receives into]─> TRANSACTION (txid) [as output_address]

TRANSACTION (txid)
  │
  ├── has many input_addresses[] (with input_amounts[])
  ├── has many output_addresses[] (with output_amounts[])
  ├── has a timestamp
  └──[observed relayed by]──> IP (src_ip → dst_ip, with src_port/dst_port)

IP
  └──[resolves to]──> geo_country, asn (via GeoIP DB lookup)
```

> **Important distinction:** A Bitcoin transaction, as recorded on-chain, contains no IP address field at all — inputs, outputs, amounts, and a signature only. `src_ip`/`dst_ip` are **P2P network-layer observations**: which peer was seen relaying the transaction to which other peer, gathered by monitoring the gossip/broadcast layer. A peer relaying a transaction is not necessarily its originator or owner — nodes forward transactions for others as part of normal propagation. So the IP↔wallet link is a **probabilistic correlation** (e.g., first-seen timing, propagation consistency), never a proven ownership claim.

## 4. Graph Model (what you'll actually build in NetworkX)

**Node types:**
- `wallet` node — one per unique address
- `tx` node — one per unique txid
- `ip` node — one per unique IP seen

**Edge types:**
- `wallet --spends_in--> tx` (input relationship, weighted by input_amount)
- `tx --pays_to--> wallet` (output relationship, weighted by output_amount)
- `tx --observed_relayed_by--> ip` (network correlation only — timing-based probabilistic link, not ownership)

This is the exact structure your Day 4–5 (NetworkX + CIOH clustering) work will build on — every later feature (entity clustering, peeling-chain traversal, risk propagation) walks this same graph.

## 5. Notes / Open Decisions
- Decide the timestamp granularity your generator will use (second vs millisecond) — affects geo-velocity anomaly detection later.
- Decide whether `is_change_output` will be inferred at generation time (ground truth) or purely detected later (for evaluation purposes) — recommend generating it as ground truth so you can measure clustering accuracy.
