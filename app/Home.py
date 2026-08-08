"""Fraudlens — Dashboard.

Entry point of the Streamlit app: headline fraud metrics across the full
transaction graph. This is "Step 1" of the demo story -- from here, use the
sidebar to open Fraud Network or Transaction Explorer.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from fraudlens.config import sync_streamlit_secrets_to_env

sync_streamlit_secrets_to_env()

from fraudlens.ui.components import empty_state, load_data_service, style_transactions_table
from fraudlens.ui.theme import configure_page

configure_page("Dashboard")

st.title("🕸️ Fraudlens")
st.caption(
    "Graph-native fraud analytics on CognoDB — XGBoost flags the transaction, "
    "the graph reveals who else is involved."
)

service = load_data_service()
if service is None:
    st.stop()

with st.spinner("Loading dashboard..."):
    stats = service.dashboard_stats()
    by_category = service.fraud_by_category()
    alerts = service.recent_high_risk_transactions(limit=8)

if stats["total_transactions"] == 0:
    empty_state("No transactions found yet. Run the seed script to load data.", icon="📭")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{stats['total_transactions']:,}")
col2.metric(
    "Confirmed fraud",
    f"{stats['fraud_count']:,}",
    f"{stats['fraud_count'] / stats['total_transactions']:.2%} of all txns",
)
col3.metric("Fraud amount", f"${stats['fraud_amount']:,.0f}")
col4.metric("Customers · Merchants", f"{stats['customers']:,} · {stats['merchants']:,}")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("XGBoost risk signal")
    if stats["avg_fraud_probability"] is None:
        empty_state(
            "No model scores yet. Run `scripts/train_model.py` then "
            "`scripts/score_transactions.py`.",
            icon="🤖",
        )
    else:
        m1, m2 = st.columns(2)
        m1.metric("Avg. predicted risk", f"{stats['avg_fraud_probability']:.1%}")
        m2.metric("High-risk (≥ 50%)", f"{stats['high_risk_count']:,}")
        st.caption(
            "XGBoost scores every transaction independently. Explore *why* a score is "
            "high in Fraud Network and Transaction Explorer."
        )

with col_b:
    st.subheader("Fraud by category")
    if not by_category:
        empty_state("No fraud recorded yet.", icon="📊")
    else:
        cat_df = pd.DataFrame(by_category)
        fig = px.bar(
            cat_df, x="category", y="fraud_count", color="fraud_amount",
            color_continuous_scale="Reds", labels={"fraud_count": "fraud transactions"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Highest-risk transactions right now")
if not alerts:
    empty_state("No scored transactions yet.", icon="🚨")
else:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(style_transactions_table(alerts_df), width="stretch", hide_index=True)

st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/1_Fraud_Network.py", label="Explore the Fraud Network →", icon="🕸️")
with nav2:
    st.page_link("pages/2_Transaction_Explorer.py", label="Open Transaction Explorer →", icon="🔍")
