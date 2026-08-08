#!/usr/bin/env python
"""CLI entry point: train the fraud-risk XGBoost model on the generated dataset.

Usage:
    python scripts/train_model.py

Requires data/generated/transactions.csv to exist (run scripts/generate_data.py first).
"""
from __future__ import annotations

import sys

from fraudlens import config
from fraudlens.ml.train import load_transactions, save_artifacts, train_model


def main() -> None:
    if not config.TRANSACTIONS_CSV.exists():
        print("No generated dataset found. Run scripts/generate_data.py first.")
        sys.exit(1)

    print("Loading transactions...")
    transactions = load_transactions()

    print("Engineering features and training XGBoost model...")
    model, metrics = train_model(transactions)
    save_artifacts(model, metrics)

    print(f"  train rows:        {metrics.n_train:,}")
    print(f"  test rows:         {metrics.n_test:,}")
    print(f"  fraud rate:        {metrics.fraud_rate:.2%}")
    print(f"  ROC AUC:           {metrics.roc_auc:.4f}")
    print(f"  Average precision: {metrics.average_precision:.4f}")
    print(f"  Precision@0.5:     {metrics.precision_at_0_5:.4f}")
    print(f"  Recall@0.5:        {metrics.recall_at_0_5:.4f}")
    print(f"  F1@0.5:            {metrics.f1_at_0_5:.4f}")
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
