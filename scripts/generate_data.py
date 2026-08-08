#!/usr/bin/env python
"""CLI entry point: generate the synthetic Fraudlens dataset.

Usage:
    python scripts/generate_data.py
"""
from __future__ import annotations

from fraudlens import config
from fraudlens.data.generator import generate_all, save_dataset


def main() -> None:
    print(f"Generating synthetic dataset (seed={config.RANDOM_SEED})...")
    dataset = generate_all()
    save_dataset(dataset)

    fraud_count = int(dataset.transactions["is_fraud"].sum())
    total = len(dataset.transactions)
    print(f"  customers:    {len(dataset.customers):>7,}")
    print(f"  merchants:    {len(dataset.merchants):>7,}")
    print(f"  categories:   {len(dataset.categories):>7,}")
    print(f"  devices:      {len(dataset.devices):>7,}")
    print(f"  transactions: {total:>7,}  (fraud: {fraud_count:,} = {fraud_count / total:.2%})")
    print(f"Saved to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
