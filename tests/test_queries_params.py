"""Sanity checks that db.queries functions are properly parameterized.

Uses a fake connection instead of a live CognoDB instance so these run
anywhere. The point is to prove query strings never interpolate user input
directly -- everything goes through the parameters dict.
"""
from __future__ import annotations

from fraudlens.db import queries


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run_query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        self.calls.append((cypher, parameters or {}))
        return []

    def run_write(self, cypher: str, parameters: dict | None = None) -> dict:
        self.calls.append((cypher, parameters or {}))
        return {}

    def run_write_batch(self, cypher: str, rows: list[dict], batch_param_name: str = "rows") -> dict:
        self.calls.append((cypher, {batch_param_name: rows}))
        return {}


def test_customer_network_passes_parameters_not_string_concatenation():
    conn = FakeConnection()
    queries.customer_network(conn, "CUST00001", limit=50)
    cypher, params = conn.calls[-1]
    assert params == {"customer_id": "CUST00001", "limit": 50}
    assert "CUST00001" not in cypher
    assert "$customer_id" in cypher


def test_search_transactions_never_concatenates_search_string():
    conn = FakeConnection()
    malicious = "'; MATCH (n) DETACH DELETE n; //"
    queries.search_transactions(conn, search=malicious, limit=10)
    cypher, params = conn.calls[-1]
    assert malicious not in cypher
    assert params["search"] == malicious


def test_update_fraud_probabilities_uses_unwind_batch():
    conn = FakeConnection()
    rows = [{"transaction_id": "T1", "fraud_probability": 0.9}]
    queries.update_fraud_probabilities(conn, rows)
    cypher, params = conn.calls[-1]
    assert "UNWIND" in cypher
    assert params["rows"] == rows


def test_customer_network_is_a_multi_hop_traversal():
    cypher = queries._CUSTOMER_NETWORK
    # Customer -> Transaction -> Merchant -> Category is 3 relationship hops.
    assert cypher.count("-[:") >= 3


def test_shared_device_rings_query_has_no_string_concatenated_params():
    conn = FakeConnection()
    queries.shared_device_rings(conn, limit=5)
    cypher, params = conn.calls[-1]
    assert params == {"limit": 5}
