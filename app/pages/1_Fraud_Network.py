"""Fraudlens — Fraud Network.

Explore the graph directly: pick a customer to see everything connected to
them (a 3-hop traversal), see which merchants are most connected to fraud,
see what other customers touch a suspicious merchant, and see device-sharing
fraud rings -- the query a relational schema would find awkward.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudlens.config import sync_streamlit_secrets_to_env

sync_streamlit_secrets_to_env()

from fraudlens.ui.components import empty_state, load_data_service, style_transactions_table
from fraudlens.ui.graph_view import render_customer_network, render_ring_graph
from fraudlens.ui.theme import configure_page

configure_page("Fraud Network", icon="🕸️")
st.title("🕸️ Fraud Network")
st.caption("Everything connected to a customer, merchant, or shared device — the relationships behind the risk score.")

service = load_data_service()
if service is None:
    st.stop()

tab_customer, tab_merchant, tab_rings = st.tabs(
    ["Customer network", "Merchants connected to fraud", "Shared-device rings"]
)

# --- Tab 1: customer network (multi-hop) ------------------------------------
with tab_customer:
    st.subheader("What is this customer connected to?")
    search = st.text_input("Search customers by name or ID", key="cust_search")
    with st.spinner("Loading customers..."):
        customers = service.list_customers(search=search or None, limit=25)

    if not customers:
        empty_state("No customers match your search.", icon="🔎")
    else:
        options = {
            f"{c['name']} ({c['customer_id']}) · {int(c['fraud_count'])} flagged txns": c["customer_id"]
            for c in customers
        }
        choice = st.selectbox("Customer (ranked by flagged transactions)", list(options.keys()))
        customer_id = options[choice]

        with st.spinner("Traversing Customer → Transaction → Merchant → Category..."):
            network_rows = service.customer_network(customer_id, limit=150)

        if not network_rows:
            empty_state("This customer has no transactions yet.", icon="🗂️")
        else:
            graph_html = render_customer_network(network_rows)
            if graph_html:
                st.iframe(graph_html, height=580)
            st.caption(
                "★ customer · ● transaction (red = confirmed fraud, orange = high XGBoost risk) "
                "· ■ merchant (red = compromised) · ▲ category"
            )
            table_df = pd.DataFrame(network_rows)[
                ["transaction_id", "amount", "timestamp", "merchant_name", "category",
                 "channel", "is_fraud", "fraud_probability"]
            ]
            st.dataframe(style_transactions_table(table_df), width="stretch", hide_index=True)

# --- Tab 2: merchants connected to fraud, and who else touches them --------
with tab_merchant:
    st.subheader("Which merchants are connected to fraudulent activity?")
    with st.spinner("Aggregating fraud by merchant..."):
        risky_merchants = service.merchants_connected_to_fraud(limit=15)

    if not risky_merchants:
        empty_state("No fraud recorded yet.", icon="📊")
    else:
        merchants_df = pd.DataFrame(risky_merchants).rename(
            columns={"connected_customers": "distinct customers touched"}
        )
        st.dataframe(merchants_df, width="stretch", hide_index=True)

        options = {
            f"{m['name']} ({m['merchant_id']}) · {m['fraud_txn_count']} fraud txns": m["merchant_id"]
            for m in risky_merchants
        }
        choice = st.selectbox("Inspect a merchant: who else is connected?", list(options.keys()))
        merchant_id = options[choice]

        with st.spinner("Finding other customers connected to this merchant's suspicious activity..."):
            connected = service.merchant_connected_customers(merchant_id, limit=25)

        if not connected:
            empty_state("No other customers connected to suspicious activity at this merchant.", icon="🗂️")
        else:
            st.dataframe(pd.DataFrame(connected), width="stretch", hide_index=True)

# --- Tab 3: shared-device rings (relational-awkward query) -----------------
with tab_rings:
    st.subheader("Devices shared across otherwise-unrelated customers")
    st.caption(
        "A relational schema finds this awkward: it needs a self-join over transactions "
        "grouped by device, filtered to groups with more than one customer and at least one "
        "fraud flag — and following the ring one hop further needs another hand-written join. "
        "Here it's a single pattern match."
    )
    with st.spinner("Finding shared-device rings..."):
        rings = service.shared_device_rings(limit=10)

    if not rings:
        empty_state("No shared-device rings detected in this dataset.", icon="🔗")
    else:
        options = {
            f"{r['device_id']} · {r['customer_count']} customers · {r['fraud_txn_count']} fraud txns": r
            for r in rings
        }
        choice = st.selectbox("Ring", list(options.keys()))
        ring = options[choice]

        with st.spinner("Expanding ring..."):
            expansion = service.ring_expansion(ring["device_id"])

        customers_for_graph = [{"customer_id": cid, "name": cid} for cid in ring["customer_ids"]]
        graph_html = render_ring_graph(ring["device_id"], customers_for_graph, expansion)
        if graph_html:
            st.iframe(graph_html, height=480)

        st.write("**Customers sharing this device:**", ", ".join(ring["customer_ids"]))
        if expansion:
            st.write("**Other devices these customers also used** (extending the ring one more hop):")
            st.dataframe(pd.DataFrame(expansion), width="stretch", hide_index=True)
