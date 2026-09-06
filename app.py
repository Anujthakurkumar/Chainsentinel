"""
app.py

Day 12 build — Streamlit dashboard: the ranked alert table + basic layout.
This is the first piece of the visual deliverable the problem statement
requires ("dashboard/visualization showing flagged entities and evidence for
each flag"). Day 13 adds the link-analysis graph view on top of this.

Run with:  streamlit run app.py
(See the chat for exact Colab commands, since Colab needs a tunnel to view
a live Streamlit server.)
"""

import pickle

import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Config

from graph_view import build_subgraph

st.set_page_config(page_title="ChainSentinel", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("final_alerts_explained.csv")
    return df


@st.cache_resource
def load_graph():
    with open("btc_graph_wallet_only.gpickle", "rb") as f:
        return pickle.load(f)


df = load_data()
W = load_graph()

# ----------------------------
# Header + summary metrics
# ----------------------------
st.title("🔍 ChainSentinel")
st.caption("AI-Powered Bitcoin Transaction Monitoring — offline prototype with ranked, explainable risk alerts")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Wallets analyzed", f"{len(df):,}")
col2.metric("Flagged (risk ≥ 0.5)", f"{(df['final_risk'] >= 0.5).sum():,}")
col3.metric("High confidence (risk ≥ 0.9)", f"{(df['final_risk'] >= 0.9).sum():,}")
avg_risk_flagged = df.loc[df["final_risk"] >= 0.5, "final_risk"].mean()
col4.metric("Avg risk (flagged)", f"{avg_risk_flagged:.2f}" if pd.notna(avg_risk_flagged) else "—")

st.divider()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.5, 0.05)

pattern_options = ["All"] + sorted(
    {p for plist in df["pattern_types_involved"].dropna() for p in plist.split(",") if p}
)
pattern_filter = st.sidebar.selectbox("Pattern type involved", pattern_options)

search_wallet = st.sidebar.text_input("Search wallet address (partial match)")

show_seeds_only = st.sidebar.checkbox("Show only seed wallets (known threat intel)")

# ----------------------------
# Apply filters
# ----------------------------
filtered = df[df["final_risk"] >= risk_threshold].copy()

if pattern_filter != "All":
    filtered = filtered[filtered["pattern_types_involved"].fillna("").str.contains(pattern_filter)]

if search_wallet:
    filtered = filtered[filtered["wallet"].str.contains(search_wallet, case=False, na=False)]

if show_seeds_only and "is_seed" in filtered.columns:
    filtered = filtered[filtered["is_seed"] == True]  # noqa: E712

filtered = filtered.sort_values(["final_risk", "tie_break_score"], ascending=[False, False])

# ----------------------------
# Pattern distribution chart
# ----------------------------
st.subheader("Pattern types among flagged wallets")
pattern_counts = {}
for plist in filtered["pattern_types_involved"].dropna():
    for p in plist.split(","):
        if p:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
if pattern_counts:
    st.bar_chart(pd.Series(pattern_counts, name="count"))
else:
    st.caption("No pattern-tagged wallets in the current filter.")

st.divider()

# ----------------------------
# Ranked alert table
# ----------------------------
st.subheader(f"Ranked alerts ({len(filtered)} wallets match current filters)")

display_cols = ["wallet", "final_risk", "risk_propagated", "risk_anomaly", "risk_peeling",
                 "pattern_types_involved", "final_explanation"]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].rename(columns={
        "wallet": "Wallet",
        "final_risk": "Risk Score",
        "risk_propagated": "Network Risk",
        "risk_anomaly": "Anomaly Risk",
        "risk_peeling": "Peeling Risk",
        "pattern_types_involved": "Patterns",
        "final_explanation": "Why Flagged",
    }),
    use_container_width=True,
    height=500,
)

st.caption(
    "Risk = combined via noisy-OR from three independent signals: graph-based risk "
    "propagation from known cash-out wallets, Isolation Forest anomaly detection, and "
    "a Random Forest peeling-chain classifier. See 'Why Flagged' for SHAP-based, "
    "per-feature reasons behind each score."
)

st.divider()

# ----------------------------
# Day 13: Link-analysis graph view with drill-down
# ----------------------------
st.subheader("🕸️ Link Analysis Graph")
st.caption(
    "Shows a FOCUSED neighborhood, not the whole dataset (~2000 nodes would be an "
    "unreadable blob). Nodes: circles = wallets (colored green→red by risk score, "
    "thick border = seed wallet), diamonds = transactions (dark = illicit pattern). "
    "Click any node to see its details below."
)

graph_col1, graph_col2 = st.columns([1, 2])
with graph_col1:
    view_mode = st.radio("View", ["Top riskiest wallets", "Search a specific wallet"])
    if view_mode == "Top riskiest wallets":
        top_k = st.slider("Number of top-risk wallets", 3, 30, 12)
        center_wallet = None
    else:
        top_k = 12
        center_wallet = st.text_input("Wallet address (exact match)")

nodes, edges, node_info = build_subgraph(df, W, top_k=top_k, center_wallet=center_wallet or None)

config = Config(
    width=900,
    height=550,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#f0f000",
    collapsible=False,
)

selected_node = agraph(nodes=nodes, edges=edges, config=config)

st.markdown("#### Node details")
if selected_node and selected_node in node_info:
    info = node_info[selected_node]
    if info["type"] == "wallet":
        st.markdown(f"**Wallet:** `{selected_node}`")
        st.markdown(f"**Risk score:** {info['risk']:.3f}")
        st.markdown(f"**Patterns involved:** {info['patterns'] or 'None'}")
        st.markdown(f"**Explanation:** {info['explanation']}")
    else:
        st.markdown(f"**Transaction:** `{selected_node}`")
        st.markdown(f"**Pattern type:** {info['pattern_type']}")
        st.markdown(f"**Timestamp:** {info['timestamp']}")
        st.markdown(f"**Fee:** {info['fee']}")
else:
    st.caption("Click a node in the graph above to see its details here.")
