"""Fraudlens — Transaction Explorer.

Search and inspect individual transactions: amount, merchant, customer,
XGBoost risk, and fraud status, plus the graph context around each one.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudlens.config import sync_streamlit_secrets_to_env

sync_streamlit_secrets_to_env()

from fraudlens.ui.components import empty_state, load_data_service, risk_pill, style_transactions_table
from fraudlens.ui.theme import configure_page

configure_page("Transaction Explorer", icon="🔍")
st.title("🔍 Transaction Explorer")
st.caption("Every transaction, with its XGBoost risk score and fraud status.")

service = load_data_service()
if service is None:
    st.stop()

with st.form("txn_filters"):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    search = c1.text_input("Search (transaction ID, customer, or merchant)")
    min_amount = c2.number_input("Min amount ($)", min_value=0.0, value=0.0, step=50.0)
    min_risk = c3.slider("Min XGBoost risk", 0.0, 1.0, 0.0, 0.05)
    fraud_only = c4.checkbox("Confirmed fraud only")
    st.form_submit_button("Search", width="stretch")

with st.spinner("Searching transactions..."):
    results = service.search_transactions(
        search=search or None,
        min_amount=min_amount or None,
        fraud_only=fraud_only,
        min_risk=min_risk or None,
        limit=200,
    )

if not results:
    empty_state("No transactions match these filters.", icon="🔎")
    st.stop()

st.caption(f"{len(results)} transaction(s) shown (max 200).")
results_df = pd.DataFrame(results)
st.dataframe(style_transactions_table(results_df), width="stretch", hide_index=True, height=340)

st.divider()
st.subheader("Inspect a transaction")
tx_options = {
    f"{r['transaction_id']} · ${r['amount']:,.2f} · {r['customer_name']} → {r['merchant_name']}": r["transaction_id"]
    for r in results
}
tx_choice = st.selectbox("Transaction", list(tx_options.keys()))
transaction_id = tx_options[tx_choice]

with st.spinner("Loading transaction detail..."):
    detail = service.transaction_detail(transaction_id)

if detail is None:
    empty_state("Transaction not found.", icon="🗂️")
else:
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"### {detail['transaction_id']}")
        st.markdown(f"**${detail['amount']:,.2f}** · {detail['timestamp']} · {detail['channel']}")

        risk_badge = risk_pill(detail["fraud_probability"])
        if detail["is_fraud"]:
            fraud_badge = (
                '<span class="fraudlens-pill" style="background:#d6454522; color:#d64545;">'
                "Confirmed fraud</span>"
            )
        else:
            fraud_badge = (
                '<span class="fraudlens-pill" style="background:#2f9e5b22; color:#2f9e5b;">'
                "Not flagged</span>"
            )
        st.markdown(f"{risk_badge}&nbsp;&nbsp;{fraud_badge}", unsafe_allow_html=True)

        st.markdown("**Customer**")
        st.write(f"{detail['customer_name']} ({detail['customer_id']}) · {detail['customer_city']}")

        st.markdown("**Merchant**")
        merchant_flag = " 🚩 compromised" if detail["merchant_compromised"] else ""
        st.write(
            f"{detail['merchant_name']} ({detail['merchant_id']}) · "
            f"{detail['merchant_city']} · {detail['category']}{merchant_flag}"
        )

        st.markdown("**Device**")
        st.write(f"{detail['device_id']} ({detail['device_type']})")

    with right:
        risk_display = "—" if detail["fraud_probability"] is None else f"{detail['fraud_probability']:.1%}"
        st.metric("XGBoost fraud probability", risk_display)
        st.metric("Other customers on this device", detail["device_shared_with_customers"])
        if detail["device_shared_with_customers"] > 0:
            st.warning(
                "This device has been used by other customers too — see the Fraud Network's "
                "shared-device rings view."
            )

    st.divider()
    st.info(
        "**XGBoost identifies transaction-level risk, while CognoDB reveals the relationships "
        "behind that risk.** The probability above comes from the model; the customer, merchant, "
        "category, and device connections come straight from the graph."
    )
