#!/usr/bin/env python
"""CLI entry point: score every transaction in CognoDB with the trained model.

Usage:
    python scripts/score_transactions.py

Requires a trained model (run scripts/train_model.py first) and a seeded
CognoDB instance (run scripts/seed_database.py first).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fraudlens import config
from fraudlens.db.connection import DatabaseUnavailableError, get_connection
from fraudlens.ml.score import load_model, score_all_transactions, write_scores


def main() -> None:
    if not config.MODEL_PATH.exists():
        print("No trained model found. Run scripts/train_model.py first.")
        sys.exit(1)

    model = load_model()

    try:
        with get_connection() as conn:
            print("Fetching graph-derived features from CognoDB...")
            scored = score_all_transactions(conn, model)
            if scored.empty:
                print("No transactions found in CognoDB. Run scripts/seed_database.py first.")
                sys.exit(1)

            print(f"Scoring {len(scored):,} transactions and writing fraud_probability back...")
            written = write_scores(conn, scored)

            high_risk = int((scored["fraud_probability"] >= 0.5).sum())
            print(f"Updated {written:,} transactions. {high_risk:,} flagged high-risk (>= 0.5).")
    except DatabaseUnavailableError as exc:
        print(f"CognoDB is not reachable: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
