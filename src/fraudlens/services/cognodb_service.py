"""Live data service backed by a real CognoDB connection.

Thin adapter: every method just forwards to the matching function in
db.queries with the connection already bound, so the UI layer never has to
know about the driver, sessions, or Cypher.
"""
from __future__ import annotations

from fraudlens.db import queries
from fraudlens.db.connection import CognoDBConnection
from fraudlens.config import get_cognodb_settings


class CognoDBService:
    def __init__(self) -> None:
        self._conn = CognoDBConnection(get_cognodb_settings())
        self._conn.connect()

    @property
    def is_live(self) -> bool:
        return True

    def close(self) -> None:
        self._conn.close()

    def dashboard_stats(self) -> dict:
        return queries.dashboard_stats(self._conn)

    def fraud_by_category(self) -> list[dict]:
        return queries.fraud_by_category(self._conn)

    def recent_high_risk_transactions(self, limit: int = 10) -> list[dict]:
        return queries.recent_high_risk_transactions(self._conn, limit)

    def merchants_connected_to_fraud(self, limit: int = 20) -> list[dict]:
        return queries.merchants_connected_to_fraud(self._conn, limit)

    def customer_network(self, customer_id: str, limit: int = 200) -> list[dict]:
        return queries.customer_network(self._conn, customer_id, limit)

    def merchant_connected_customers(
        self, merchant_id: str, risk_threshold: float = 0.5, limit: int = 50
    ) -> list[dict]:
        return queries.merchant_connected_customers(self._conn, merchant_id, risk_threshold, limit)

    def shared_device_rings(self, limit: int = 15) -> list[dict]:
        return queries.shared_device_rings(self._conn, limit)

    def ring_expansion(self, device_id: str) -> list[dict]:
        return queries.ring_expansion(self._conn, device_id)

    def list_customers(self, search: str | None = None, limit: int = 50) -> list[dict]:
        return queries.list_customers(self._conn, search, limit)

    def list_merchants(self, search: str | None = None, limit: int = 50) -> list[dict]:
        return queries.list_merchants(self._conn, search, limit)

    def transaction_detail(self, transaction_id: str) -> dict | None:
        return queries.transaction_detail(self._conn, transaction_id)

    def search_transactions(
        self,
        search: str | None = None,
        min_amount: float | None = None,
        fraud_only: bool = False,
        min_risk: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        return queries.search_transactions(self._conn, search, min_amount, fraud_only, min_risk, limit)
