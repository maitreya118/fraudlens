"""Reusable Streamlit building blocks: status banners, empty states, styling."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudlens.config import get_cognodb_settings, is_demo_mode
from fraudlens.db.connection import DatabaseUnavailableError
from fraudlens.services.data_service import get_data_service
from fraudlens.services.demo_service import DemoDatasetMissingError
from fraudlens.ui.theme import NEUTRAL, RISK_HIGH, RISK_LOW, RISK_MEDIUM


def render_demo_banner() -> None:
    st.markdown(
        """
        <div class="fraudlens-banner fraudlens-banner-demo">
        ⚠️ <b>Demo Mode</b> — not connected to CognoDB. Showing a local synthetic dataset
        (pandas over generated CSVs), not live graph queries. Set <code>COGNODB_URI</code>,
        <code>COGNODB_USER</code> and <code>COGNODB_PASSWORD</code> to use the real graph database.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_not_configured_banner() -> None:
    st.markdown(
        """
        <div class="fraudlens-banner fraudlens-banner-error">
        🔌 <b>CognoDB connection is not configured.</b> Database-backed features are unavailable
        until <code>COGNODB_URI</code>, <code>COGNODB_USER</code> and <code>COGNODB_PASSWORD</code>
        are set (see <code>.env.example</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_connection_error_banner(message: str) -> None:
    st.markdown(
        f"""
        <div class="fraudlens-banner fraudlens-banner-error">
        🔌 <b>Could not reach CognoDB.</b> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_dataset_missing_banner() -> None:
    st.markdown(
        """
        <div class="fraudlens-banner fraudlens-banner-error">
        📂 <b>No local demo dataset found.</b> Run
        <code>python scripts/generate_data.py</code> to create one, or configure CognoDB.
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "🗂️") -> None:
    st.markdown(
        f"""
        <div class="fraudlens-banner fraudlens-banner-info" style="text-align:center; padding:2.2rem 1rem;">
        <div style="font-size:2rem;">{icon}</div>
        <div style="margin-top:0.4rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _cached_service():
    return get_data_service()


def load_data_service():
    """Return a working data service, or render the right error/empty state and return None.

    Centralizes the fallback story required by the assignment: never crash,
    always tell the user plainly whether they're looking at CognoDB or a
    local demo dataset, and never blur the two together.
    """
    try:
        service = _cached_service()
    except DatabaseUnavailableError as exc:
        if not get_cognodb_settings().is_configured:
            render_not_configured_banner()
        else:
            render_connection_error_banner(str(exc))
        return None
    except DemoDatasetMissingError:
        render_demo_dataset_missing_banner()
        return None

    if not service.is_live:
        render_demo_banner()
    return service


def risk_label(probability: float | None) -> tuple[str, str]:
    """Map a fraud probability to a (label, color) pair for badges."""
    if probability is None or pd.isna(probability):
        return "Not scored", NEUTRAL
    if probability >= 0.7:
        return "High risk", RISK_HIGH
    if probability >= 0.3:
        return "Medium risk", RISK_MEDIUM
    return "Low risk", RISK_LOW


def risk_pill(probability: float | None) -> str:
    label, color = risk_label(probability)
    pct = "" if probability is None or pd.isna(probability) else f" ({probability:.0%})"
    return (
        f'<span class="fraudlens-pill" style="background:{color}22; color:{color};">'
        f"{label}{pct}</span>"
    )


def style_transactions_table(df: pd.DataFrame):
    """Pandas Styler that color-codes fraud status and risk for st.dataframe."""

    def _highlight_fraud(row: pd.Series) -> list[str]:
        if row.get("is_fraud") == 1:
            return ["background-color: rgba(214, 69, 69, 0.12)"] * len(row)
        return [""] * len(row)

    def _risk_shade(value: float) -> str:
        if pd.isna(value):
            return ""
        alpha = 0.08 + min(max(value, 0.0), 1.0) * 0.35
        return f"background-color: rgba(214, 69, 69, {alpha:.2f})"

    styler = df.style.apply(_highlight_fraud, axis=1)
    if "fraud_probability" in df.columns:
        styler = styler.map(_risk_shade, subset=["fraud_probability"])
    if "amount" in df.columns:
        styler = styler.format({"amount": "${:,.2f}"})
    if "fraud_probability" in df.columns:
        styler = styler.format({"fraud_probability": "{:.0%}"}, na_rep="—")
    return styler
