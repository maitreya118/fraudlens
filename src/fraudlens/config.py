"""Centralized configuration for Fraudlens.

All secrets and tunables are read from the environment (populated from a
local .env file during development, or from Streamlit secrets when deployed).
Nothing here is hardcoded or committed as a real credential.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "generated"
MODELS_DIR = PROJECT_ROOT / "models"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
MERCHANTS_CSV = DATA_DIR / "merchants.csv"
CATEGORIES_CSV = DATA_DIR / "categories.csv"
DEVICES_CSV = DATA_DIR / "devices.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"

MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_SPEC_PATH = MODELS_DIR / "feature_spec.json"

# --- Synthetic dataset sizing --------------------------------------------
# The demo story targets ~25,000 transactions across a realistic customer /
# merchant population, with a handful of injected fraud rings.
NUM_CUSTOMERS = 800
NUM_MERCHANTS = 150
NUM_DEVICES = 1200
NUM_TRANSACTIONS = 25_000
NUM_FRAUD_RING_DEVICES = 18
NUM_COMPROMISED_MERCHANTS = 6
RANDOM_SEED = 42

# --- CognoDB connection ---------------------------------------------------


@dataclass(frozen=True)
class CognoDBSettings:
    """Connection settings for the CognoDB (Neo4j-compatible) instance."""

    uri: str
    user: str
    password: str

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.user and self.password)


def get_cognodb_settings() -> CognoDBSettings:
    """Read CognoDB connection settings from the environment.

    Reads fresh from os.environ on every call (rather than caching at import
    time) so that Streamlit secrets synced later in the process are picked up.
    """
    return CognoDBSettings(
        uri=os.getenv("COGNODB_URI", "").strip(),
        user=os.getenv("COGNODB_USER", "cognodb").strip(),
        password=os.getenv("COGNODB_PASSWORD", "").strip(),
    )


def is_demo_mode() -> bool:
    """True when the app should use the local synthetic dataset instead of CognoDB.

    Demo mode activates automatically whenever CognoDB credentials are missing
    (so the app never crashes), or when explicitly forced with
    FRAUDLENS_DEMO_MODE=true for local UI development.
    """
    forced = os.getenv("FRAUDLENS_DEMO_MODE", "false").strip().lower() == "true"
    return forced or not get_cognodb_settings().is_configured


def sync_streamlit_secrets_to_env() -> None:
    """Copy Streamlit-deployed secrets into os.environ.

    Lets the rest of the codebase read configuration uniformly via
    os.getenv/get_cognodb_settings, regardless of whether it is running
    locally (.env) or on Streamlit Community Cloud (st.secrets). No-ops
    silently when no secrets.toml is present, e.g. local CLI scripts.
    """
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            os.environ.setdefault(key, str(value))
    except Exception:
        pass
