#!/usr/bin/env python
"""CLI entry point: load the generated dataset into CognoDB.

Usage:
    python scripts/seed_database.py

Requires COGNODB_URI / COGNODB_USER / COGNODB_PASSWORD in the environment
(see .env.example) and a generated dataset (run scripts/generate_data.py first).
Safe to re-run: all writes are MERGE-based and idempotent.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from fraudlens import config
from fraudlens.db.connection import DatabaseUnavailableError, get_connection
from fraudlens.db.schema import apply_schema
from fraudlens.db.seed import (
    seed_categories,
    seed_customers,
    seed_devices,
    seed_merchants,
    seed_transactions,
)


def main() -> None:
    missing = [
        p.name for p in [
            config.CUSTOMERS_CSV, config.MERCHANTS_CSV, config.CATEGORIES_CSV,
            config.DEVICES_CSV, config.TRANSACTIONS_CSV,
        ] if not p.exists()
    ]
    if missing:
        print(f"Missing generated data files: {missing}. Run scripts/generate_data.py first.")
        sys.exit(1)

    customers = pd.read_csv(config.CUSTOMERS_CSV)
    merchants = pd.read_csv(config.MERCHANTS_CSV)
    categories = pd.read_csv(config.CATEGORIES_CSV)
    devices = pd.read_csv(config.DEVICES_CSV)
    transactions = pd.read_csv(config.TRANSACTIONS_CSV)

    try:
        with get_connection() as conn:
            print("Applying schema (constraints + indexes)...")
            apply_schema(conn)

            print("Seeding graph (MERGE-based, safe to re-run)...")
            start = time.time()
            seed_categories(conn, categories)
            seed_customers(conn, customers)
            seed_merchants(conn, merchants)
            seed_devices(conn, devices)
            seed_transactions(conn, transactions)
            elapsed = time.time() - start
            print(f"Done in {elapsed:.1f}s.")
    except DatabaseUnavailableError as exc:
        print(f"CognoDB is not reachable: {exc}")
        print("Check COGNODB_URI / COGNODB_USER / COGNODB_PASSWORD in your .env file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
