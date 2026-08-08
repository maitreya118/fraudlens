"""Score every transaction in CognoDB with the trained XGBoost model.

Unlike training (which reads the CSV dataset directly), scoring pulls its
features straight from the graph: per-customer spend stats, merchant fraud
rates, first-merchant-visit timestamps, and -- the graph-native one --
how many distinct customers share each device. Those are joined client-side
into the same FEATURE_COLUMNS used at training time, then written back onto
each Transaction node.
"""
from __future__ import annotations

import joblib
import pandas as pd

from fraudlens import config
from fraudlens.db import queries
from fraudlens.db.connection import CognoDBConnection
from fraudlens.ml.features import select_feature_matrix

FETCH_BATCH_SIZE = 5000
WRITE_BATCH_SIZE = 1000


def load_model():
    return joblib.load(config.MODEL_PATH)


def _fetch_all_transactions(conn: CognoDBConnection) -> pd.DataFrame:
    total = queries.count_transactions(conn)
    frames: list[pd.DataFrame] = []
    skip = 0
    while skip < total:
        batch = queries.fetch_raw_transactions_batch(conn, skip, FETCH_BATCH_SIZE)
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        skip += FETCH_BATCH_SIZE
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_scoring_features(conn: CognoDBConnection) -> pd.DataFrame:
    """Rebuild FEATURE_COLUMNS for every transaction currently stored in CognoDB."""
    txns = _fetch_all_transactions(conn)
    if txns.empty:
        return txns

    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns["hour_of_day"] = txns["timestamp"].dt.hour
    txns["day_of_week"] = txns["timestamp"].dt.dayofweek
    txns["is_weekend"] = (txns["day_of_week"] >= 5).astype(int)
    txns["channel_online"] = (txns["channel"] == "online").astype(int)

    cust_stats = pd.DataFrame(queries.fetch_customer_amount_stats(conn))
    txns = txns.merge(cust_stats, on="customer_id", how="left")
    txns["std_amount"] = txns["std_amount"].replace(0, pd.NA)
    txns["amount_zscore_for_customer"] = (
        (txns["amount"] - txns["mean_amount"]) / txns["std_amount"]
    ).astype(float).fillna(0.0)

    merchant_rates = pd.DataFrame(queries.fetch_merchant_fraud_rates(conn))
    txns = txns.merge(merchant_rates, on="merchant_id", how="left")
    txns["merchant_fraud_rate"] = txns["fraud_rate"].fillna(0.0)

    device_counts = pd.DataFrame(queries.fetch_device_shared_counts(conn))
    txns = txns.merge(device_counts, on="device_id", how="left")
    txns["device_shared_customer_count"] = txns["shared_customer_count"].fillna(1).astype(int)

    first_visit = pd.DataFrame(queries.fetch_first_merchant_visit(conn))
    txns = txns.merge(first_visit, on=["customer_id", "merchant_id"], how="left")
    txns["first_seen"] = pd.to_datetime(txns["first_seen"])
    txns["is_new_merchant_for_customer"] = (txns["timestamp"] <= txns["first_seen"]).astype(int)

    return txns


def score_all_transactions(conn: CognoDBConnection, model) -> pd.DataFrame:
    features_df = build_scoring_features(conn)
    if features_df.empty:
        return features_df
    X = select_feature_matrix(features_df)
    features_df["fraud_probability"] = model.predict_proba(X)[:, 1]
    return features_df


def write_scores(conn: CognoDBConnection, scored_df: pd.DataFrame) -> int:
    rows = scored_df[["transaction_id", "fraud_probability"]].to_dict("records")
    for i in range(0, len(rows), WRITE_BATCH_SIZE):
        queries.update_fraud_probabilities(conn, rows[i : i + WRITE_BATCH_SIZE])
    queries.update_customer_risk_scores(conn)
    return len(rows)
