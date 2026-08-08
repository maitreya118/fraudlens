"""Shared visual constants and page setup for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

APP_TITLE = "Fraudlens"
APP_ICON = "🕸️"

RISK_HIGH = "#d64545"
RISK_MEDIUM = "#e0a030"
RISK_LOW = "#2f9e5b"
NEUTRAL = "#6b7280"


def configure_page(title: str, icon: str = APP_ICON) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} · {APP_TITLE}", page_icon=icon, layout="wide")
    _inject_css()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        div[data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.06);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 10px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }
        .fraudlens-banner {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 1.2rem;
            font-size: 0.92rem;
            line-height: 1.4;
        }
        .fraudlens-banner-demo {
            background: rgba(224, 160, 48, 0.12);
            border: 1px solid rgba(224, 160, 48, 0.45);
        }
        .fraudlens-banner-error {
            background: rgba(214, 69, 69, 0.10);
            border: 1px solid rgba(214, 69, 69, 0.45);
        }
        .fraudlens-banner-info {
            background: rgba(76, 110, 245, 0.10);
            border: 1px solid rgba(76, 110, 245, 0.35);
        }
        .fraudlens-pill {
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
