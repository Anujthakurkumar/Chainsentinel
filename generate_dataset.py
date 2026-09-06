"""
generate_dataset.py

Day 3 build — completes the synthetic (Bitcoin-SHAPED, not real chain data)
dataset generator by injecting illicit patterns on top of the same UTXO/entity
ledger from Day 2, and wiring in GeoIP via geoip_lookup.py.

Patterns injected, each interleaved at random points across the normal
transaction timeline (NOT appended at the end — a detector that just flags
"the last N rows" would otherwise trivially "solve" this dataset):

  1. Peeling chain  -> pattern_type = "peeling"
     One entity repeatedly spends its own change output, sending a small
     "peel" to a cash-out entity and the large remainder back to itself at a
     fresh address, for several hops.

  2. Fan-out / fan-in -> pattern_type = "fanout" / "fanin"
     One entity splits funds out to many entities (fan-out), which shortly
     after each send most of what they received on to one aggregator entity
     (fan-in) — classic layering.

  3. CoinJoin-like mixing -> pattern_type = "coinjoin"
     Several independent entities each contribute one input to a SINGLE
     transaction with multiple near-equal-value outputs, breaking the
     common-input-ownership assumption.

Every record carries `pattern_type` (ground truth) plus `chain_id`/`group_id`
where relevant, and `coinjoin_ground_truth`. These are generation-time facts —
keep them separate from anything your models later predict.
"""

import hashlib
import random
import string
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from geoip_lookup import get_geo, using_real_geoip, get_stats

# ----------------------------
# Config
# ----------------------------
NUM_NORMAL_ENTITIES = 60
NUM_CASHOUT_ENTITIES = 5     # simulated mule/exchange sinks, reserved id range
NUM_ENTITIES = NUM_NORMAL_ENTITIES + NUM_CASHOUT_ENTITIES

NUM_NORMAL_TX = 800
NUM_PEELING_CHAINS = 18   # bumped up from 4 for a statistically meaningful Day 9 evaluation
NUM_FANOUT_GROUPS = 4
NUM_COINJOIN_GROUPS = 5
NUM_HARD_NEGATIVE_CHAINS = 14   # deliberate structural look-alikes, labeled normal

NUM_PEER_IPS = 50
START_TIME = datetime(2026, 1, 1)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

CASHOUT_ENTITY_IDS = list(range(NUM_NORMAL_ENTITIES, NUM_ENTITIES))


# ----------------------------
# Helpers (same as Day 2, unchanged)
# ----------------------------
def random_address(prefix: str = "bc1q") -> str:
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=38))


def random_ip() -> str:
    """Generates a public-looking IPv4 address, explicitly avoiding private/reserved
    blocks (10.0.0.0/8, 172.16-31.x, 192.168.x, 127.x, 169.254.x, 0.x, 224+.x multicast)
    so it actually resolves against real GeoLite2 data instead of silently triggering
    the synthetic fallback."""
    while True:
        o1 = random.randint(1, 223)
        if o1 == 10 or o1 == 127 or o1 >= 224:
            continue
        o2 = random.randint(0, 255)
        if o1 == 172 and 16 <= o2 <= 31:
            continue
        if o1 == 192 and o2 == 168:
            continue
        if o1 == 169 and o2 == 254:
            continue
        o3 = random.randint(0, 255)
        o4 = random.randint(1, 254)
        return f"{o1}.{o2}.{o3}.{o4}"


def random_txid() -> str:
    return hashlib.sha256(str(random.random()).encode() + uuid.uuid4().bytes).hexdigest()


def sample_amount() -> float:
    return round(np.random.pareto(2.0) * 0.05 + 0.0001, 8)


# ----------------------------
# Entity + address + UTXO state
# ----------------------------
entity_addresses = {e: [random_address()] for e in range(NUM_ENTITIES)}
address_to_entity = {addrs[0]: e for e, addrs in entity_addresses.items()}
peer_ips = [random_ip() for _ in range(NUM_PEER_IPS)]
utxo_pool = {}  # (txid, vout) -> {"address":..., "amount":..., "entity":...}

# ----------------------------
# Peer-affinity model: real geographic clustering of peer IPs + per-entity home pools.
# Fixes the earlier flaw where src_ip was picked uniformly at random per transaction,
# which made geo-velocity pure noise (any two consecutive txs got unrelated global
# locations regardless of pattern type). Now each entity behaves like a node that
# consistently relays through 2-3 geographically nearby peers, with RARE, DESIGNED
# exceptions for cash-out flows and CoinJoin mixing -- exactly where real investigators
# expect to see proxy/VPN/mixing-service geographic inconsistency.
# ----------------------------
EXCEPTION_RATE_NORMAL = 0.05
EXCEPTION_RATE_CASHOUT_FLOW = 0.6
EXCEPTION_RATE_COINJOIN = 0.5

peer_ip_geo = {ip: get_geo(ip) for ip in peer_ips}
country_to_ips = defaultdict(list)
for ip, geo in peer_ip_geo.items():
    country_to_ips[geo["geo_country"]].append(ip)

# only countries with enough IPs to form a real "home pool" are usable as a home base
_viable_countries = [c for c, ips in country_to_ips.items() if len(ips) >= 2]

entity_home_pool = {}
for e in range(NUM_ENTITIES):
    if _viable_countries:
        home_country = random.choice(_viable_countries)
        pool = country_to_ips[home_country]
        entity_home_pool[e] = random.sample(pool, k=min(3, len(pool)))
    else:
        entity_home_pool[e] = random.sample(peer_ips, k=min(3, len(peer_ips)))


def seed_genesis_utxos():
    for e in range(NUM_ENTITIES):
        addr = entity_addresses[e][0]
        for _ in range(random.randint(1, 3)):
            txid = f"genesis_{e}_{random.randint(0, 999999)}"
            utxo_pool[(txid, 0)] = {"address": addr, "amount": round(np.random.pareto(1.5) * 0.5 + 0.01, 8), "entity": e}


def seed_large_utxo(entity_id: int, amount: float):
    """Bootstraps a big spendable balance for an entity — used to give peeling
    chains and fan-out events a large starting sum to work with."""
    addr = entity_addresses[entity_id][0]
    txid = f"seed_{entity_id}_{random.randint(0, 999999)}"
    utxo_pool[(txid, 0)] = {"address": addr, "amount": amount, "entity": entity_id}


seed_genesis_utxos()


def new_change_address(entity_id: int) -> str:
    addr = random_address()
    entity_addresses[entity_id].append(addr)
    address_to_entity[addr] = entity_id
    return addr


def entity_receiving_address(entity_id: int) -> str:
    if random.random() < 0.7 and entity_addresses[entity_id]:
        return random.choice(entity_addresses[entity_id])
    return new_change_address(entity_id)


def entity_utxo_keys(entity_id: int):
    return [k for k, v in utxo_pool.items() if v["entity"] == entity_id]


def entity_balance(entity_id: int) -> float:
    return sum(v["amount"] for v in utxo_pool.values() if v["entity"] == entity_id)


# ----------------------------
# Shared transaction commit helper
# ----------------------------
def commit_tx(current_time, input_keys, outputs, pattern_type, extra=None):
    """
    outputs: list of dicts {"address":..., "amount":..., "entity":..., "is_change":bool}
    Consumes input_keys from utxo_pool, creates new UTXOs for each output,
    and returns the full record dict (schema-compliant + ground truth fields).
    """
    input_addresses = [utxo_pool[k]["address"] for k in input_keys]
    input_amounts = [utxo_pool[k]["amount"] for k in input_keys]
    input_prevouts = [f"{k[0]}:{k[1]}" for k in input_keys]
    sender_entity = utxo_pool[input_keys[0]]["entity"]

    for k in input_keys:
        del utxo_pool[k]

    txid = random_txid()
    for vout_i, o in enumerate(outputs):
        utxo_pool[(txid, vout_i)] = {"address": o["address"], "amount": o["amount"], "entity": o["entity"]}

    # ---- peer-affinity src_ip selection, with designed exceptions ----
    # sender's normal behavior: relay through their own home peer pool.
    # Exception cases (cash-out flows, CoinJoin) deliberately use a random GLOBAL peer
    # instead, modeling proxy/VPN/mixing-service infrastructure -- this is what makes
    # geo-velocity a real signal instead of noise: it now spikes specifically where
    # we've designed it to, not uniformly everywhere.
    is_cashout_flow = any(o["entity"] in CASHOUT_ENTITY_IDS for o in outputs)
    if pattern_type == "coinjoin":
        exception_rate = EXCEPTION_RATE_COINJOIN
    elif is_cashout_flow:
        exception_rate = EXCEPTION_RATE_CASHOUT_FLOW
    else:
        exception_rate = EXCEPTION_RATE_NORMAL

    is_geo_exception = random.random() < exception_rate
    if is_geo_exception:
        src_ip = random.choice(peer_ips)  # distant/random peer -- the designed anomaly
    else:
        src_ip = random.choice(entity_home_pool.get(sender_entity, peer_ips))
    dst_ip = random.choice([ip for ip in peer_ips if ip != src_ip])
    src_geo = peer_ip_geo[src_ip]

    record = {
        "timestamp": current_time.isoformat(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(1024, 65535),
        "dst_port": 8333,
        "txid": txid,
        "input_addresses": input_addresses,
        "input_prevouts": input_prevouts,
        "input_amounts": input_amounts,
        "output_addresses": [o["address"] for o in outputs],
        "output_amounts": [o["amount"] for o in outputs],
        "fee": round(sum(input_amounts) - sum(o["amount"] for o in outputs), 8),
        "geo_country": src_geo["geo_country"],
        "asn": src_geo["asn"],
        "label": "normal" if pattern_type == "normal" else "illicit",
        "pattern_type": pattern_type,  # ground truth: normal | peeling | fanout | fanin | coinjoin
        "sender_entity_ground_truth": sender_entity,
        "output_entity_ground_truth": [o["entity"] for o in outputs],
        "is_change_ground_truth": [o["is_change"] for o in outputs],
        "coinjoin_ground_truth": pattern_type == "coinjoin",
        "geo_exception_ground_truth": is_geo_exception,
        "chain_id": None,
        "group_id": None,
    }
    if extra:
        record.update(extra)
    return record


# ----------------------------
# Pattern generators
# ----------------------------
def generate_normal_tx(current_time):
    entities_with_funds = list({u["entity"] for u in utxo_pool.values()})
    if not entities_with_funds:
        seed_genesis_utxos()
        entities_with_funds = list({u["entity"] for u in utxo_pool.values()})
    sender_entity = random.choice(entities_with_funds)
    keys = entity_utxo_keys(sender_entity)
    if not keys:
        return None
    n_inputs = min(random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0], len(keys))
    input_keys = random.sample(keys, n_inputs)
    total_in = sum(utxo_pool[k]["amount"] for k in input_keys)
    fee = round(total_in * random.uniform(0.0005, 0.002), 8)
    spendable = max(total_in - fee, 0.00001)
    if spendable <= 0.00001:
        return None

    receiver_entity = random.choice([e for e in range(NUM_NORMAL_ENTITIES) if e != sender_entity])
    has_change = random.random() < 0.55
    if has_change:
        split = random.uniform(0.4, 0.85)
        pay_amt, change_amt = round(spendable * split, 8), round(spendable * (1 - split), 8)
    else:
        pay_amt, change_amt = round(spendable, 8), 0.0

    outputs = [{"address": entity_receiving_address(receiver_entity), "amount": pay_amt,
                "entity": receiver_entity, "is_change": False}]
    if has_change and change_amt > 0:
        outputs.append({"address": new_change_address(sender_entity), "amount": change_amt,
                         "entity": sender_entity, "is_change": True})
    return commit_tx(current_time, input_keys, outputs, "normal")


def generate_peeling_chain(chain_id, current_time, hop_count, gap_range=(30, 300),
                            peel_fraction_range=(0.05, 0.15), style="medium"):
    """One entity peels a small amount off to a cash-out sink each hop, passing
    the bulk of the balance forward to itself, for several hops in a row.

    style/gap_range/peel_fraction_range let us generate a MIX of chain
    behaviors rather than one uniform pattern -- e.g. a tight, fast,
    automated-looking chain vs a slower, looser, more human-paced one.
    This matters for Day 9: training a classifier on only one narrow style
    of chain risks it learning "hop_count > 5" rather than genuine
    peeling-chain structure."""
    victim_entity = random.choice(range(NUM_NORMAL_ENTITIES))
    seed_large_utxo(victim_entity, amount=round(np.random.uniform(5, 20), 8))
    records = []
    for hop in range(hop_count):
        keys = entity_utxo_keys(victim_entity)
        if not keys:
            break
        input_key = max(keys, key=lambda k: utxo_pool[k]["amount"])  # spend the big one
        total_in = utxo_pool[input_key]["amount"]
        fee = round(total_in * random.uniform(0.0003, 0.001), 8)
        spendable = max(total_in - fee, 0.00001)

        peel_fraction = random.uniform(*peel_fraction_range)
        peel_amt = round(spendable * peel_fraction, 8)
        remainder_amt = round(spendable - peel_amt, 8)

        cashout_entity = random.choice(CASHOUT_ENTITY_IDS)
        outputs = [
            {"address": entity_receiving_address(cashout_entity), "amount": peel_amt,
             "entity": cashout_entity, "is_change": False},
            {"address": new_change_address(victim_entity), "amount": remainder_amt,
             "entity": victim_entity, "is_change": True},
        ]
        rec = commit_tx(current_time, [input_key], outputs, "peeling",
                         extra={"chain_id": chain_id, "hop_index": hop, "chain_style": style})
        records.append(rec)
        current_time += timedelta(seconds=random.randint(*gap_range))
        if remainder_amt < 0.02:
            break
    return records, current_time


def generate_hard_negative_chain(chain_id, current_time, hop_count, gap_range=(1800, 172800),
                                  ratio_range=(0.05, 0.3)):
    """HARD NEGATIVE: models a legitimate multi-hop payment forwarding pattern
    (e.g. an exchange hot-wallet passing funds through several intermediaries)
    that is STRUCTURALLY similar to a peeling chain -- asymmetric 2-output
    split, the larger output spent again shortly after -- but is genuinely
    NOT peeling:
      - ownership actually changes hands each hop (a different entity each
        time), unlike real peeling where ONE entity keeps control throughout
      - timing is irregular: minutes to two days, not a consistent cadence
      - the split ratio varies unpredictably hop to hop, not a controlled
        "skim fraction"
      - neither output ever goes to a cash-out/mule entity

    This is a well-documented real false-positive source for naive peeling
    detectors, and exactly what Day 9's classifier needs to see labeled
    NEGATIVE (pattern_type stays "normal") to prove it learned genuine
    peeling structure rather than just "asymmetric 2-output chain exists."
    """
    current_entity = random.choice(range(NUM_NORMAL_ENTITIES))
    seed_large_utxo(current_entity, amount=round(np.random.uniform(3, 15), 8))
    records = []
    for hop in range(hop_count):
        keys = entity_utxo_keys(current_entity)
        if not keys:
            break
        input_key = max(keys, key=lambda k: utxo_pool[k]["amount"])
        total_in = utxo_pool[input_key]["amount"]
        fee = round(total_in * random.uniform(0.0003, 0.001), 8)
        spendable = max(total_in - fee, 0.00001)

        small_frac = random.uniform(*ratio_range)
        small_amt = round(spendable * small_frac, 8)
        large_amt = round(spendable - small_amt, 8)

        other_entities = [e for e in range(NUM_NORMAL_ENTITIES) if e != current_entity]
        small_recipient = random.choice(other_entities)
        large_recipient = random.choice([e for e in other_entities if e != small_recipient])

        outputs = [
            {"address": entity_receiving_address(small_recipient), "amount": small_amt,
             "entity": small_recipient, "is_change": False},
            {"address": entity_receiving_address(large_recipient), "amount": large_amt,
             "entity": large_recipient, "is_change": False},
        ]
        # pattern_type stays "normal" -- this is a NEGATIVE example, tagged only
        # for our own tracking of how often it fools the traversal detector
        rec = commit_tx(current_time, [input_key], outputs, "normal",
                         extra={"hard_negative_chain_id": chain_id, "hop_index": hop})
        records.append(rec)
        current_time += timedelta(seconds=random.randint(*gap_range))
        current_entity = large_recipient  # ownership genuinely changes hands -- NOT peeling
        if large_amt < 0.02:
            break
    return records, current_time


def generate_fanout_fanin(group_id, current_time, k):
    """Source entity splits funds to k entities (fan-out), then each of those
    k entities forwards most of what it received to one aggregator (fan-in)."""
    records = []
    source_entity = random.choice(range(NUM_NORMAL_ENTITIES))
    seed_large_utxo(source_entity, amount=round(np.random.uniform(3, 4) * k, 8))

    keys = entity_utxo_keys(source_entity)
    input_key = max(keys, key=lambda k_: utxo_pool[k_]["amount"])
    total_in = utxo_pool[input_key]["amount"]
    fee = round(total_in * 0.001, 8)
    spendable = max(total_in - fee, 0.00001)

    recipients = random.sample([e for e in range(NUM_NORMAL_ENTITIES) if e != source_entity], k)
    share = round(spendable / k, 8)
    outputs = [{"address": entity_receiving_address(r), "amount": share, "entity": r, "is_change": False}
               for r in recipients]
    fanout_rec = commit_tx(current_time, [input_key], outputs, "fanout", extra={"group_id": group_id})
    fanout_txid = fanout_rec["txid"]
    records.append(fanout_rec)
    current_time += timedelta(seconds=random.randint(60, 600))

    aggregator_entity = random.choice([e for e in range(NUM_NORMAL_ENTITIES) if e not in recipients + [source_entity]])
    for i, r in enumerate(recipients):
        r_key = (fanout_txid, i)  # exactly the UTXO the fan-out just created for this recipient
        if r_key not in utxo_pool:
            continue  # safety: shouldn't happen, but never spend a UTXO that isn't there
        r_total = utxo_pool[r_key]["amount"]
        r_fee = round(r_total * 0.001, 8)
        forward_amt = round(max(r_total - r_fee, 0.00001) * random.uniform(0.85, 0.98), 8)
        out = [{"address": entity_receiving_address(aggregator_entity), "amount": forward_amt,
                "entity": aggregator_entity, "is_change": False}]
        rec = commit_tx(current_time, [r_key], out, "fanin", extra={"group_id": group_id})
        records.append(rec)
        current_time += timedelta(seconds=random.randint(20, 180))
    return records, current_time


def generate_coinjoin(group_id, current_time, k):
    """k independent entities each contribute one input to a SINGLE transaction
    with k near-equal-value outputs — breaks common-input-ownership clustering."""
    participants = random.sample(range(NUM_NORMAL_ENTITIES), k)
    target_amount = round(np.random.uniform(0.05, 0.3), 8)

    input_keys = []
    for p in participants:
        seed_large_utxo(p, amount=round(target_amount / 0.95 + random.uniform(0, 0.01), 8))
        keys = entity_utxo_keys(p)
        input_keys.append(max(keys, key=lambda k_: utxo_pool[k_]["amount"]))

    total_in = sum(utxo_pool[k_]["amount"] for k_ in input_keys)
    fee = round(total_in * 0.0015, 8)
    per_output = round((total_in - fee) / k, 8)

    # Each participant gets one equal-value output back at a FRESH address
    # they control — standard CoinJoin behavior.
    shuffled = participants[:]
    random.shuffle(shuffled)
    outputs = [{"address": new_change_address(p), "amount": per_output, "entity": p, "is_change": False}
               for p in shuffled]

    rec = commit_tx(current_time, input_keys, outputs, "coinjoin",
                     extra={"group_id": group_id, "mix_size": k})
    return [rec], current_time + timedelta(seconds=random.randint(10, 60))


# ----------------------------
# Main interleaved generation loop
# ----------------------------
all_records = []
current_time = START_TIME

# schedule injection points spread across the normal-transaction timeline
injection_slots = sorted(random.sample(range(20, NUM_NORMAL_TX - 20),
                                        NUM_PEELING_CHAINS + NUM_FANOUT_GROUPS + NUM_COINJOIN_GROUPS
                                        + NUM_HARD_NEGATIVE_CHAINS))
injection_plan = (
    [("peeling", i) for i in range(NUM_PEELING_CHAINS)]
    + [("fanout", i) for i in range(NUM_FANOUT_GROUPS)]
    + [("coinjoin", i) for i in range(NUM_COINJOIN_GROUPS)]
    + [("hard_negative", i) for i in range(NUM_HARD_NEGATIVE_CHAINS)]
)
random.shuffle(injection_plan)
slot_to_event = dict(zip(injection_slots, injection_plan))

PEELING_CHAIN_STYLES = {
    "tight": {"hop_range": (6, 10), "gap_range": (20, 150), "peel_fraction_range": (0.05, 0.12)},
    "medium": {"hop_range": (5, 9), "gap_range": (30, 300), "peel_fraction_range": (0.05, 0.15)},
    "loose": {"hop_range": (3, 5), "gap_range": (200, 2000), "peel_fraction_range": (0.08, 0.20)},
}

for step in range(NUM_NORMAL_TX):
    rec = generate_normal_tx(current_time)
    if rec:
        all_records.append(rec)
    current_time += timedelta(seconds=random.randint(5, 600))

    if step in slot_to_event:
        kind, idx = slot_to_event[step]
        if kind == "peeling":
            style = random.choice(list(PEELING_CHAIN_STYLES.keys()))
            cfg = PEELING_CHAIN_STYLES[style]
            recs, current_time = generate_peeling_chain(
                f"chain_{idx}", current_time,
                hop_count=random.randint(*cfg["hop_range"]),
                gap_range=cfg["gap_range"],
                peel_fraction_range=cfg["peel_fraction_range"],
                style=style,
            )
        elif kind == "fanout":
            recs, current_time = generate_fanout_fanin(f"group_{idx}", current_time,
                                                         k=random.randint(6, 12))
        elif kind == "coinjoin":
            recs, current_time = generate_coinjoin(f"mix_{idx}", current_time,
                                                     k=random.randint(3, 6))
        else:  # hard_negative
            recs, current_time = generate_hard_negative_chain(
                f"hardneg_{idx}", current_time, hop_count=random.randint(2, 6))
        all_records.extend(recs)

df = pd.DataFrame(all_records).sort_values("timestamp").reset_index(drop=True)

df_csv = df.copy()
list_cols = ["input_addresses", "input_prevouts", "input_amounts",
             "output_addresses", "output_amounts",
             "output_entity_ground_truth", "is_change_ground_truth"]
for col in list_cols:
    df_csv[col] = df_csv[col].apply(lambda x: str(x))

df.to_json("btc_dataset.json", orient="records", indent=2)
df_csv.to_csv("btc_dataset.csv", index=False)

print(f"GeoIP backend: {'REAL GeoLite2' if using_real_geoip() else 'synthetic fallback'}")
stats = get_stats()
total_lookups = stats["real"] + stats["fallback"]
if total_lookups:
    print(f"GeoIP resolution: {stats['real']} real ({stats['real']/total_lookups:.1%}), "
          f"{stats['fallback']} fallback ({stats['fallback']/total_lookups:.1%})")
n_exceptions = df["geo_exception_ground_truth"].sum() if "geo_exception_ground_truth" in df else 0
print(f"Geo-exception (distant peer) transactions: {n_exceptions} ({n_exceptions/len(df):.1%})")
if "hard_negative_chain_id" in df.columns:
    n_hardneg_tx = df["hard_negative_chain_id"].notna().sum()
    n_hardneg_chains = df["hard_negative_chain_id"].nunique()
    print(f"Hard-negative pass-through transactions: {n_hardneg_tx} across {n_hardneg_chains} chains (labeled normal)")
print(f"\nTotal transactions: {len(df)}")
print(df["pattern_type"].value_counts())
print("\nWrote: btc_dataset.json, btc_dataset.csv")
