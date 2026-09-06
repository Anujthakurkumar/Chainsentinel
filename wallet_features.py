"""
wallet_features.py

Day 6 build — computes per-wallet features to feed into the Isolation Forest
anomaly detector (Day 7). Runs on btc_graph.gpickle (the FULL graph, including
IP nodes — needed here specifically for geo-velocity, unlike Day 5's
wallet-only projection which was for clustering).

Features computed per wallet:
  avg_amount_zscore       - how atypical this wallet's average tx amount is,
                            vs the GLOBAL amount distribution across the dataset
  max_abs_amount_zscore   - the single most atypical amount this wallet was
                            ever involved in
  tx_count                - total distinct transactions (in + out) this wallet appears in
  tx_frequency_per_day    - tx_count normalized by the wallet's active time span
  fan_in_degree           - number of times this wallet RECEIVED funds (pays_to edges in)
  fan_out_degree          - number of times this wallet SPENT funds (spends_in edges out)
  max_geo_velocity_kmh    - fastest implied travel speed between two consecutive
                            peer-IP locations observed relaying this wallet's txs
                            (see caveat below — this is a NETWORK-layer signal)
  avg_geo_velocity_kmh    - average of the same, across all consecutive hops
  time_of_day_entropy     - Shannon entropy of the hour-of-day distribution of
                            this wallet's transactions (low = suspiciously regular/bot-like,
                            high = spread naturally across the day)

IMPORTANT CAVEAT (carried over from the schema doc): geo-velocity here is built
from src_ip -> observed peer location, which is a RELAY OBSERVATION, not proof
of where the wallet's owner physically is. A high geo-velocity value means "the
peers relaying this wallet's transactions were geographically inconsistent in a
way that would be physically implausible for one continuously-connected node" —
that's still a meaningful anomaly signal (e.g. proxy/VPN hopping, or multiple
distinct devices), just don't oversell it in your write-up as "the criminal
traveled at 4000 km/h."

Ground-truth columns (pattern_type involvement, entity ids) are attached for
EVALUATION ONLY on Day 7 — they must NOT be fed into the Isolation Forest as
features, since that would leak the label the model is supposed to detect.
"""

import math
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from geoip_lookup import get_geo


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def shannon_entropy(counts: list) -> float:
    """Shannon entropy (bits) of a distribution given as raw counts. 0 = all in
    one bucket (perfectly regular), higher = more spread out."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def build_wallet_features(graph_path: str = "btc_graph.gpickle") -> pd.DataFrame:
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # ---- global amount distribution, for z-scoring individual wallets against ----
    all_amounts = [d["amount"] for _, _, d in G.edges(data=True) if "amount" in d]
    global_mean, global_std = np.mean(all_amounts), np.std(all_amounts)
    global_std = global_std if global_std > 0 else 1e-9  # guard divide-by-zero

    # ---- pre-resolve geo for every IP node once (avoid repeat lookups) ----
    ip_geo_cache = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "ip":
            ip_geo_cache[n] = get_geo(n)

    wallet_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "wallet"]
    rows = []

    for w in wallet_nodes:
        amounts = []
        timestamps = []
        tx_ids = set()
        pattern_flags = set()

        # incoming (this wallet was PAID) -- edges are tx -> wallet
        for tx, _, d in G.in_edges(w, data=True):
            if d.get("relation") != "pays_to":
                continue
            amounts.append(d["amount"])
            tx_ids.add(tx)
            tx_data = G.nodes[tx]
            timestamps.append(tx_data["timestamp"])
            if tx_data.get("pattern_type") not in (None, "normal"):
                pattern_flags.add(tx_data["pattern_type"])

        fan_in_degree = len(amounts)

        # outgoing (this wallet SPENT) -- edges are wallet -> tx
        out_amounts = []
        for _, tx, d in G.out_edges(w, data=True):
            if d.get("relation") != "spends_in":
                continue
            out_amounts.append(d["amount"])
            tx_ids.add(tx)
            tx_data = G.nodes[tx]
            timestamps.append(tx_data["timestamp"])
            if tx_data.get("pattern_type") not in (None, "normal"):
                pattern_flags.add(tx_data["pattern_type"])

        fan_out_degree = len(out_amounts)
        amounts.extend(out_amounts)

        if not amounts:
            continue  # shouldn't happen, but skip defensively

        avg_amount = float(np.mean(amounts))
        avg_amount_zscore = (avg_amount - global_mean) / global_std
        max_abs_amount_zscore = float(max(abs((a - global_mean) / global_std) for a in amounts))

        tx_count = len(tx_ids)
        ts_parsed = sorted(datetime.fromisoformat(t) for t in timestamps)
        span_days = max((ts_parsed[-1] - ts_parsed[0]).total_seconds() / 86400.0, 1e-6)
        tx_frequency_per_day = tx_count / span_days if len(ts_parsed) > 1 else float(tx_count)

        hour_counts = [0] * 24
        for t in ts_parsed:
            hour_counts[t.hour] += 1
        time_of_day_entropy = shannon_entropy(hour_counts)

        # ---- geo-velocity: walk this wallet's transactions in time order, using
        # each tx's src_ip (the peer OBSERVED relaying it) as a location sample ----
        geo_points = []
        for tx in tx_ids:
            tx_data = G.nodes[tx]
            # find this tx's src_ip via its observed_relayed_by edge
            for _, ip_node, d in G.out_edges(tx, data=True):
                if d.get("relation") == "observed_relayed_by" and ip_node in ip_geo_cache:
                    geo_points.append((tx_data["timestamp"], ip_geo_cache[ip_node]))
                    break
        geo_points.sort(key=lambda x: x[0])

        velocities = []
        for (t1, g1), (t2, g2) in zip(geo_points, geo_points[1:]):
            dt_hours = (datetime.fromisoformat(t2) - datetime.fromisoformat(t1)).total_seconds() / 3600.0
            if dt_hours <= 1e-6:
                continue  # same-instant or bad ordering, skip to avoid divide-by-zero blowup
            dist_km = haversine_km(g1["lat"], g1["lon"], g2["lat"], g2["lon"])
            velocities.append(dist_km / dt_hours)

        max_geo_velocity_kmh = float(max(velocities)) if velocities else 0.0
        avg_geo_velocity_kmh = float(np.mean(velocities)) if velocities else 0.0

        rows.append({
            "wallet": w,
            "avg_amount_zscore": avg_amount_zscore,
            "max_abs_amount_zscore": max_abs_amount_zscore,
            "tx_count": tx_count,
            "tx_frequency_per_day": tx_frequency_per_day,
            "fan_in_degree": fan_in_degree,
            "fan_out_degree": fan_out_degree,
            "max_geo_velocity_kmh": max_geo_velocity_kmh,
            "avg_geo_velocity_kmh": avg_geo_velocity_kmh,
            "time_of_day_entropy": time_of_day_entropy,
            # ---- ground truth, for Day 7 evaluation ONLY -- never train on these ----
            "entity_ground_truth": G.nodes[w].get("entity_ground_truth"),
            "involved_in_illicit_pattern": bool(pattern_flags),
            "pattern_types_involved": ",".join(sorted(pattern_flags)) if pattern_flags else "",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_wallet_features("btc_graph.gpickle")
    print(f"Computed features for {len(df)} wallets")
    print(f"Wallets touching an illicit pattern: {df['involved_in_illicit_pattern'].sum()} "
          f"({df['involved_in_illicit_pattern'].mean():.1%})")
    print()

    feature_cols = ["avg_amount_zscore", "max_abs_amount_zscore", "tx_count",
                     "tx_frequency_per_day", "fan_in_degree", "fan_out_degree",
                     "max_geo_velocity_kmh", "avg_geo_velocity_kmh", "time_of_day_entropy"]

    print("=== Feature summary: normal wallets ===")
    print(df[~df["involved_in_illicit_pattern"]][feature_cols].describe().loc[["mean", "std", "max"]].to_string())
    print()
    print("=== Feature summary: wallets touched by an illicit pattern ===")
    print(df[df["involved_in_illicit_pattern"]][feature_cols].describe().loc[["mean", "std", "max"]].to_string())

    df.to_csv("wallet_features.csv", index=False)
    print("\nSaved wallet_features.csv")
