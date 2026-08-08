"""Tests for the local, CognoDB-free DemoDataService fallback."""
from __future__ import annotations

import pytest

from fraudlens import config
from fraudlens.services.demo_service import DemoDataService, DemoDatasetMissingError


def test_missing_dataset_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRANSACTIONS_CSV", tmp_path / "does_not_exist.csv")
    with pytest.raises(DemoDatasetMissingError):
        DemoDataService()


def test_demo_service_basic_shapes(tiny_dataset):
    service = DemoDataService()
    assert service.is_live is False

    stats = service.dashboard_stats()
    assert stats["total_transactions"] == len(tiny_dataset.transactions)
    assert stats["customers"] == len(tiny_dataset.customers)
    assert stats["fraud_count"] == int(tiny_dataset.transactions["is_fraud"].sum())

    customers = service.list_customers(limit=5)
    assert len(customers) <= 5

    merchants = service.list_merchants(limit=5)
    assert len(merchants) <= 5


def test_customer_network_only_returns_that_customers_rows(tiny_dataset):
    service = DemoDataService()
    customer_id = tiny_dataset.customers.iloc[0]["customer_id"]
    rows = service.customer_network(customer_id)
    assert all(row["customer_id"] == customer_id for row in rows)


def test_shared_device_rings_only_include_multi_customer_fraud_devices(tiny_dataset):
    service = DemoDataService()
    rings = service.shared_device_rings(limit=10)
    for ring in rings:
        assert ring["customer_count"] > 1
        assert ring["fraud_txn_count"] > 0


def test_transaction_detail_roundtrip(tiny_dataset):
    service = DemoDataService()
    any_txn_id = tiny_dataset.transactions.iloc[0]["transaction_id"]
    detail = service.transaction_detail(any_txn_id)
    assert detail is not None
    assert detail["transaction_id"] == any_txn_id


def test_transaction_detail_missing_returns_none(tiny_dataset):
    service = DemoDataService()
    assert service.transaction_detail("TXN_DOES_NOT_EXIST") is None
