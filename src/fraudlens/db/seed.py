"""Idempotent batch loader: CSV files -> CognoDB graph.

Every node type is loaded with MERGE keyed on its natural identifier
(customer_id, merchant_id, category name, device_id, transaction_id), and
every relationship is created with MERGE too, so running this script twice
never produces duplicate nodes or duplicate edges. Writes are batched with
UNWIND rather than one round trip per row.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from fraudlens.db.connection import CognoDBConnection

BATCH_SIZE = 1000

_SEED_CATEGORIES = """
UNWIND $rows AS row
MERGE (cat:Category {name: row.name})
"""

_SEED_CUSTOMERS = """
UNWIND $rows AS row
MERGE (c:Customer {customer_id: row.customer_id})
SET c.name = row.name, c.email = row.email, c.city = row.city, c.signup_date = row.signup_date
"""

_SEED_MERCHANTS = """
UNWIND $rows AS row
MERGE (m:Merchant {merchant_id: row.merchant_id})
SET m.name = row.name, m.city = row.city, m.is_compromised = row.is_compromised
WITH m, row
MATCH (cat:Category {name: row.category})
MERGE (m)-[:IN_CATEGORY]->(cat)
"""

_SEED_DEVICES = """
UNWIND $rows AS row
MERGE (d:Device {device_id: row.device_id})
SET d.device_type = row.device_type
"""

_SEED_TRANSACTIONS = """
UNWIND $rows AS row
MERGE (t:Transaction {transaction_id: row.transaction_id})
SET t.amount = row.amount, t.timestamp = row.timestamp, t.channel = row.channel,
    t.is_fraud = row.is_fraud
WITH t, row
MATCH (c:Customer {customer_id: row.customer_id})
MATCH (m:Merchant {merchant_id: row.merchant_id})
MATCH (d:Device {device_id: row.device_id})
MERGE (c)-[:MADE]->(t)
MERGE (t)-[:AT_MERCHANT]->(m)
MERGE (t)-[:USED_DEVICE]->(d)
"""


def _chunks(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _seed_in_batches(
    conn: CognoDBConnection, cypher: str, df: pd.DataFrame, label: str, batch_size: int = BATCH_SIZE
) -> int:
    rows = df.to_dict("records")
    for batch in _chunks(rows, batch_size):
        conn.run_write_batch(cypher, batch)
    print(f"  seeded {len(rows):,} {label}")
    return len(rows)


def seed_categories(conn: CognoDBConnection, df: pd.DataFrame) -> int:
    return _seed_in_batches(conn, _SEED_CATEGORIES, df, "categories")


def seed_customers(conn: CognoDBConnection, df: pd.DataFrame) -> int:
    return _seed_in_batches(conn, _SEED_CUSTOMERS, df, "customers")


def seed_merchants(conn: CognoDBConnection, df: pd.DataFrame) -> int:
    return _seed_in_batches(conn, _SEED_MERCHANTS, df, "merchants")


def seed_devices(conn: CognoDBConnection, df: pd.DataFrame) -> int:
    return _seed_in_batches(conn, _SEED_DEVICES, df, "devices")


def seed_transactions(conn: CognoDBConnection, df: pd.DataFrame) -> int:
    return _seed_in_batches(conn, _SEED_TRANSACTIONS, df, "transactions")
