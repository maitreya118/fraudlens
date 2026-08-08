"""CognoDB schema: uniqueness constraints and indexes.

Uniqueness constraints double as indexes for their property and are what
make the seed script's MERGE-based loading both correct (no duplicate
Customer/Merchant/Category/Device/Transaction nodes on re-run) and fast.
The extra indexes below speed up the investigation queries the UI runs.
"""
from __future__ import annotations

from fraudlens.db.connection import CognoDBConnection

CONSTRAINTS = [
    "CREATE CONSTRAINT customer_id_unique IF NOT EXISTS "
    "FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
    "CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS "
    "FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE",
    "CREATE CONSTRAINT category_name_unique IF NOT EXISTS "
    "FOR (cat:Category) REQUIRE cat.name IS UNIQUE",
    "CREATE CONSTRAINT device_id_unique IF NOT EXISTS "
    "FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
    "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS "
    "FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX transaction_fraud_probability IF NOT EXISTS "
    "FOR (t:Transaction) ON (t.fraud_probability)",
    "CREATE INDEX transaction_is_fraud IF NOT EXISTS "
    "FOR (t:Transaction) ON (t.is_fraud)",
    "CREATE INDEX transaction_timestamp IF NOT EXISTS "
    "FOR (t:Transaction) ON (t.timestamp)",
]


def apply_schema(conn: CognoDBConnection) -> None:
    """Create all constraints and indexes. Safe to run repeatedly."""
    for statement in CONSTRAINTS + INDEXES:
        conn.run_write(statement)
