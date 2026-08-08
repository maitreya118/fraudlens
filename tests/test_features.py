"""Tests for shared feature engineering (fraudlens.ml.features)."""
from __future__ import annotations

import pandas as pd
import pytest

from fraudlens.ml.features import FEATURE_COLUMNS, build_training_features, select_feature_matrix


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame([
        {"transaction_id": "T1", "customer_id": "C1", "merchant_id": "M1", "device_id": "D1",
         "amount": 100.0, "timestamp": "2024-01-01T10:00:00", "channel": "online", "is_fraud": 0},
        {"transaction_id": "T2", "customer_id": "C1", "merchant_id": "M1", "device_id": "D1",
         "amount": 5000.0, "timestamp": "2024-01-02T02:00:00", "channel": "online", "is_fraud": 1},
        {"transaction_id": "T3", "customer_id": "C2", "merchant_id": "M1", "device_id": "D1",
         "amount": 80.0, "timestamp": "2024-01-03T14:00:00", "channel": "in_store", "is_fraud": 0},
    ])


def test_build_training_features_has_all_expected_columns():
    df = build_training_features(_sample_transactions())
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_device_shared_customer_count_counts_distinct_customers():
    df = build_training_features(_sample_transactions())
    # D1 is used by both C1 and C2 -> every row should see 2 distinct customers.
    assert (df["device_shared_customer_count"] == 2).all()


def test_is_new_merchant_flag_only_true_on_first_visit():
    df = build_training_features(_sample_transactions())
    c1_rows = df[df["customer_id"] == "C1"].sort_values("timestamp")
    assert c1_rows["is_new_merchant_for_customer"].tolist() == [1, 0]


def test_select_feature_matrix_raises_on_missing_columns():
    with pytest.raises(ValueError):
        select_feature_matrix(pd.DataFrame({"amount": [1.0]}))


def test_select_feature_matrix_returns_only_feature_columns():
    df = build_training_features(_sample_transactions())
    matrix = select_feature_matrix(df)
    assert list(matrix.columns) == FEATURE_COLUMNS
