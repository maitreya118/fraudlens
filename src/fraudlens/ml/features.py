"""Feature engineering shared by model training and live scoring.

Two callers build the same feature columns two different ways:

- Training (fraudlens.ml.train) computes them in pandas over the full
  generated dataset.
- Scoring (scripts/score_transactions.py) computes them via a Cypher query
  that walks the graph directly in CognoDB (see db/queries.py::
  fetch_scoring_features), then hands the resulting rows to
  select_feature_matrix() here.

FEATURE_COLUMNS is the single source of truth for what the model expects,
so the two paths cannot silently drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "channel_online",
    "amount_zscore_for_customer",
    "is_new_merchant_for_customer",
    "merchant_fraud_rate",
    "device_shared_customer_count",
]


def build_training_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute FEATURE_COLUMNS (+ is_fraud) over the full transactions table.

    `transactions` must have: transaction_id, customer_id, merchant_id,
    device_id, amount, timestamp, channel, is_fraud.
    """
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["channel_online"] = (df["channel"] == "online").astype(int)

    # Amount z-score relative to this customer's own spending history.
    cust_stats = df.groupby("customer_id")["amount"].agg(mean="mean", std="std")
    df = df.join(cust_stats, on="customer_id")
    df["std"] = df["std"].replace(0, np.nan)
    df["amount_zscore_for_customer"] = ((df["amount"] - df["mean"]) / df["std"]).fillna(0.0)
    df = df.drop(columns=["mean", "std"])

    # Is this the first time this customer has transacted with this merchant?
    first_seen = df.groupby(["customer_id", "merchant_id"])["timestamp"].transform("min")
    df["is_new_merchant_for_customer"] = (df["timestamp"] == first_seen).astype(int)

    # Merchant's historical fraud rate, leave-one-out to avoid trivial leakage.
    fraud_sum = df.groupby("merchant_id")["is_fraud"].transform("sum")
    count = df.groupby("merchant_id")["is_fraud"].transform("count")
    denom = (count - 1).replace(0, np.nan)
    df["merchant_fraud_rate"] = ((fraud_sum - df["is_fraud"]) / denom).fillna(0.0)

    # Graph-native feature: how many distinct customers share this device?
    df["device_shared_customer_count"] = df.groupby("device_id")["customer_id"].transform("nunique")

    return df


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order FEATURE_COLUMNS from a dataframe, failing loudly if any are missing."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return df[FEATURE_COLUMNS].astype(float)
