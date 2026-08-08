"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from fraudlens import config
from fraudlens.data.generator import GeneratedDataset, generate_all, save_dataset


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch) -> GeneratedDataset:
    """Generate a small, fast synthetic dataset into a temp dir for tests."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CUSTOMERS_CSV", tmp_path / "customers.csv")
    monkeypatch.setattr(config, "MERCHANTS_CSV", tmp_path / "merchants.csv")
    monkeypatch.setattr(config, "CATEGORIES_CSV", tmp_path / "categories.csv")
    monkeypatch.setattr(config, "DEVICES_CSV", tmp_path / "devices.csv")
    monkeypatch.setattr(config, "TRANSACTIONS_CSV", tmp_path / "transactions.csv")
    monkeypatch.setattr(config, "NUM_CUSTOMERS", 30)
    monkeypatch.setattr(config, "NUM_MERCHANTS", 10)
    monkeypatch.setattr(config, "NUM_DEVICES", 25)
    monkeypatch.setattr(config, "NUM_TRANSACTIONS", 400)
    monkeypatch.setattr(config, "NUM_FRAUD_RING_DEVICES", 3)
    monkeypatch.setattr(config, "NUM_COMPROMISED_MERCHANTS", 2)

    dataset = generate_all(seed=7)
    save_dataset(dataset)
    return dataset
