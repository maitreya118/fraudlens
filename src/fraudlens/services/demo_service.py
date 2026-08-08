"""Local, CognoDB-free data service backed by the generated CSV dataset.

Used only when CognoDB credentials are missing (automatic fallback so the
app never crashes) or FRAUDLENS_DEMO_MODE is forced for local UI work. It
answers the same questions as CognoDBService with plain pandas joins over
the flat files -- it does not simulate or pretend to run Cypher queries.
The UI is responsible for showing a clear "Demo Mode" banner whenever this
service is in use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens import config


class DemoDatasetMissingError(RuntimeError):
    """Raised when the local demo CSVs have not been generated yet."""


class DemoDataService:
    def __init__(self) -> None:
        if not config.TRANSACTIONS_CSV.exists():
            raise DemoDatasetMissingError(
                "No local demo dataset found. Run `python scripts/generate_data.py` first."
            )
        self._customers = pd.read_csv(config.CUSTOMERS_CSV)
        self._merchants = pd.read_csv(config.MERCHANTS_CSV)
        self._devices = pd.read_csv(config.DEVICES_CSV)
        self._transactions = pd.read_csv(config.TRANSACTIONS_CSV)
        self._transactions["timestamp"] = pd.to_datetime(self._transactions["timestamp"])
        self._transactions["fraud_probability"] = self._score_locally()
        self._wide = self._build_wide_table()

    @property
    def is_live(self) -> bool:
        return False

    def close(self) -> None:
        pass

    # --- setup ---------------------------------------------------------

    def _score_locally(self) -> pd.Series:
        """If a trained model is available locally, score the demo dataset with it."""
        if not config.MODEL_PATH.exists():
            return pd.Series([np.nan] * len(self._transactions))

        import joblib

        from fraudlens.ml.features import build_training_features, select_feature_matrix

        model = joblib.load(config.MODEL_PATH)
        features_df = build_training_features(self._transactions)
        X = select_feature_matrix(features_df)
        proba = model.predict_proba(X)[:, 1]
        by_id = pd.Series(proba, index=features_df["transaction_id"])
        return by_id.reindex(self._transactions["transaction_id"]).reset_index(drop=True)

    def _build_wide_table(self) -> pd.DataFrame:
        df = self._transactions.merge(
            self._customers[["customer_id", "name", "city"]].rename(
                columns={"name": "customer_name", "city": "customer_city"}
            ),
            on="customer_id",
            how="left",
        )
        df = df.merge(
            self._merchants[["merchant_id", "name", "city", "category", "is_compromised"]].rename(
                columns={"name": "merchant_name", "city": "merchant_city"}
            ),
            on="merchant_id",
            how="left",
        )
        df = df.merge(self._devices, on="device_id", how="left")
        return df

    # --- dashboard -------------------------------------------------------

    def dashboard_stats(self) -> dict:
        t = self._transactions
        fraud_mask = t["is_fraud"] == 1
        has_scores = t["fraud_probability"].notna().any()
        return {
            "total_transactions": int(len(t)),
            "fraud_count": int(fraud_mask.sum()),
            "fraud_amount": float(t.loc[fraud_mask, "amount"].sum()),
            "total_amount": float(t["amount"].sum()),
            "avg_fraud_probability": float(t["fraud_probability"].mean()) if has_scores else None,
            "high_risk_count": int((t["fraud_probability"] >= 0.5).sum()) if has_scores else 0,
            "customers": int(self._customers["customer_id"].nunique()),
            "merchants": int(self._merchants["merchant_id"].nunique()),
            "devices": int(self._devices["device_id"].nunique()),
        }

    def fraud_by_category(self) -> list[dict]:
        fraud = self._wide[self._wide["is_fraud"] == 1]
        grouped = (
            fraud.groupby("category")
            .agg(fraud_count=("transaction_id", "count"), fraud_amount=("amount", "sum"))
            .reset_index()
            .sort_values("fraud_count", ascending=False)
        )
        return grouped.to_dict("records")

    def recent_high_risk_transactions(self, limit: int = 10) -> list[dict]:
        df = self._wide.dropna(subset=["fraud_probability"])
        df = df.sort_values("fraud_probability", ascending=False).head(limit)
        cols = ["transaction_id", "amount", "timestamp", "fraud_probability", "is_fraud",
                "customer_id", "customer_name", "merchant_id", "merchant_name"]
        return _stringify_timestamps(df[cols]).to_dict("records")

    # --- Q1: merchants connected to fraud ---------------------------------

    def merchants_connected_to_fraud(self, limit: int = 20) -> list[dict]:
        fraud = self._wide[self._wide["is_fraud"] == 1]
        grouped = (
            fraud.groupby(["merchant_id", "merchant_name", "merchant_city", "is_compromised"])
            .agg(
                fraud_txn_count=("transaction_id", "count"),
                fraud_amount=("amount", "sum"),
                connected_customers=("customer_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"merchant_name": "name", "merchant_city": "city"})
            .sort_values("fraud_txn_count", ascending=False)
            .head(limit)
        )
        return grouped.to_dict("records")

    # --- Q2: customer network (multi-hop) ---------------------------------

    def customer_network(self, customer_id: str, limit: int = 200) -> list[dict]:
        df = self._wide[self._wide["customer_id"] == customer_id]
        df = df.sort_values("timestamp", ascending=False).head(limit)
        df = df.rename(columns={"is_compromised": "merchant_compromised"})
        cols = ["customer_id", "customer_name", "customer_city", "transaction_id", "amount",
                "timestamp", "is_fraud", "fraud_probability", "channel", "merchant_id",
                "merchant_name", "merchant_compromised", "category", "device_id"]
        return _stringify_timestamps(df[cols]).to_dict("records")

    # --- Q3: customers connected to a suspicious merchant ------------------

    def merchant_connected_customers(
        self, merchant_id: str, risk_threshold: float = 0.5, limit: int = 50
    ) -> list[dict]:
        df = self._wide[self._wide["merchant_id"] == merchant_id]
        suspicious = df[(df["is_fraud"] == 1) | (df["fraud_probability"] >= risk_threshold)]
        grouped = (
            suspicious.groupby(["customer_id", "customer_name", "customer_city"])
            .agg(suspicious_txn_count=("transaction_id", "count"), suspicious_amount=("amount", "sum"))
            .reset_index()
            .rename(columns={"customer_name": "name", "customer_city": "city"})
            .sort_values("suspicious_txn_count", ascending=False)
            .head(limit)
        )
        return grouped.to_dict("records")

    # --- shared-device rings ------------------------------------------------

    def shared_device_rings(self, limit: int = 15) -> list[dict]:
        grouped = self._transactions.groupby("device_id").agg(
            customer_count=("customer_id", "nunique"), fraud_txn_count=("is_fraud", "sum")
        )
        rings = grouped[(grouped["customer_count"] > 1) & (grouped["fraud_txn_count"] > 0)]
        rings = rings.sort_values(["fraud_txn_count", "customer_count"], ascending=False).head(limit)

        results = []
        for device_id, row in rings.iterrows():
            customer_ids = sorted(
                self._transactions.loc[self._transactions["device_id"] == device_id, "customer_id"]
                .unique()
                .tolist()
            )
            results.append({
                "device_id": device_id,
                "customer_count": int(row["customer_count"]),
                "fraud_txn_count": int(row["fraud_txn_count"]),
                "customer_ids": customer_ids,
            })
        return results

    def ring_expansion(self, device_id: str) -> list[dict]:
        ring_customers = self._transactions.loc[
            self._transactions["device_id"] == device_id, "customer_id"
        ].unique()
        others = self._transactions[
            self._transactions["customer_id"].isin(ring_customers)
            & (self._transactions["device_id"] != device_id)
        ]
        grouped = others.groupby("customer_id")["device_id"].agg(lambda s: sorted(set(s))).reset_index()
        grouped = grouped.rename(columns={"device_id": "other_devices"})
        grouped = grouped.merge(self._customers[["customer_id", "name"]], on="customer_id", how="left")
        return grouped[["customer_id", "name", "other_devices"]].to_dict("records")

    # --- browsing / search --------------------------------------------------

    def list_customers(self, search: str | None = None, limit: int = 50) -> list[dict]:
        grouped = self._transactions.groupby("customer_id").agg(
            txn_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
            avg_risk=("fraud_probability", "mean"),
        ).reset_index()
        merged = self._customers.merge(grouped, on="customer_id", how="left")
        merged[["txn_count", "fraud_count"]] = merged[["txn_count", "fraud_count"]].fillna(0)
        if search:
            s = search.lower()
            merged = merged[
                merged["name"].str.lower().str.contains(s) | merged["customer_id"].str.lower().str.contains(s)
            ]
        merged = merged.sort_values(["fraud_count", "avg_risk"], ascending=False).head(limit)
        return merged[["customer_id", "name", "city", "txn_count", "fraud_count", "avg_risk"]].to_dict("records")

    def list_merchants(self, search: str | None = None, limit: int = 50) -> list[dict]:
        grouped = self._transactions.groupby("merchant_id").agg(
            txn_count=("transaction_id", "count"), fraud_count=("is_fraud", "sum")
        ).reset_index()
        merged = self._merchants.merge(grouped, on="merchant_id", how="left")
        merged[["txn_count", "fraud_count"]] = merged[["txn_count", "fraud_count"]].fillna(0)
        if search:
            s = search.lower()
            merged = merged[
                merged["name"].str.lower().str.contains(s) | merged["merchant_id"].str.lower().str.contains(s)
            ]
        merged = merged.sort_values("fraud_count", ascending=False).head(limit)
        return merged[["merchant_id", "name", "city", "is_compromised", "txn_count", "fraud_count"]].to_dict("records")

    def transaction_detail(self, transaction_id: str) -> dict | None:
        matches = self._wide[self._wide["transaction_id"] == transaction_id]
        if matches.empty:
            return None
        row = matches.iloc[0]
        device_id = row["device_id"]
        shared = self._transactions[
            (self._transactions["device_id"] == device_id)
            & (self._transactions["customer_id"] != row["customer_id"])
        ]["customer_id"].nunique()
        return {
            "customer_id": row["customer_id"],
            "customer_name": row["customer_name"],
            "customer_city": row["customer_city"],
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "timestamp": str(row["timestamp"]),
            "channel": row["channel"],
            "is_fraud": int(row["is_fraud"]),
            "fraud_probability": None if pd.isna(row["fraud_probability"]) else float(row["fraud_probability"]),
            "merchant_id": row["merchant_id"],
            "merchant_name": row["merchant_name"],
            "merchant_city": row["merchant_city"],
            "merchant_compromised": bool(row["is_compromised"]),
            "category": row["category"],
            "device_id": device_id,
            "device_type": row["device_type"],
            "device_shared_with_customers": int(shared),
        }

    def search_transactions(
        self,
        search: str | None = None,
        min_amount: float | None = None,
        fraud_only: bool = False,
        min_risk: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        df = self._wide
        if min_amount is not None:
            df = df[df["amount"] >= min_amount]
        if fraud_only:
            df = df[df["is_fraud"] == 1]
        if min_risk is not None:
            df = df[df["fraud_probability"] >= min_risk]
        if search:
            s = search.lower()
            df = df[
                df["transaction_id"].str.lower().str.contains(s)
                | df["customer_name"].str.lower().str.contains(s)
                | df["merchant_name"].str.lower().str.contains(s)
            ]
        df = df.sort_values("timestamp", ascending=False).head(limit)
        cols = ["transaction_id", "amount", "timestamp", "is_fraud", "fraud_probability", "channel",
                "customer_id", "customer_name", "merchant_id", "merchant_name"]
        return _stringify_timestamps(df[cols]).to_dict("records")


def _stringify_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    return df
