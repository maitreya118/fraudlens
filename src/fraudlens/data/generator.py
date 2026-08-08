"""Synthetic dataset generator for Fraudlens.

Produces a realistic-looking transaction network: customers, merchants,
categories and devices, plus ~25,000 transactions. Fraud is not sprinkled in
at random — it is injected as two structural patterns that mirror how real
fraud rings behave, which is exactly what makes the graph queries in this
project meaningful:

1. Device-sharing rings: a handful of devices are used by several unrelated
   customers in quick, high-value bursts (classic account-takeover / mule
   pattern).
2. Compromised merchants: a handful of merchants have an elevated fraud rate
   across many different customers (classic point-of-compromise pattern).

A small amount of organic, unstructured fraud is layered on top so the
downstream XGBoost model has to learn real signal rather than memorize the
injected structures.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from fraudlens import config

CATEGORIES = [
    "Electronics",
    "Grocery",
    "Travel",
    "Dining",
    "Fashion",
    "Health & Pharmacy",
    "Home & Garden",
    "Entertainment",
    "Utilities",
    "Automotive",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
    "Seattle", "Denver",
]

# (low, high) typical transaction amount per category, in USD.
CATEGORY_AMOUNT_RANGE = {
    "Electronics": (40, 1200),
    "Grocery": (8, 150),
    "Travel": (80, 2500),
    "Dining": (10, 120),
    "Fashion": (15, 400),
    "Health & Pharmacy": (5, 200),
    "Home & Garden": (15, 600),
    "Entertainment": (8, 150),
    "Utilities": (20, 300),
    "Automotive": (30, 1500),
}

# Categories more commonly transacted online (used to bias channel choice).
ONLINE_LEANING_CATEGORIES = {"Electronics", "Travel", "Entertainment", "Fashion"}

DEVICE_TYPES = ["mobile", "web", "pos"]
DEVICE_TYPE_WEIGHTS = [0.55, 0.30, 0.15]

WINDOW_DAYS = 365


@dataclass
class GeneratedDataset:
    customers: pd.DataFrame
    merchants: pd.DataFrame
    categories: pd.DataFrame
    devices: pd.DataFrame
    transactions: pd.DataFrame


def _sample_amount(rng: np.random.Generator, category: str, elevated: bool = False) -> float:
    """Draw a plausible transaction amount for a category from a log-normal curve."""
    low, high = CATEGORY_AMOUNT_RANGE[category]
    mid = (low + high) / 2
    sigma = 0.55
    value = rng.lognormal(mean=np.log(mid), sigma=sigma)
    if elevated:
        value *= rng.uniform(1.8, 4.0)
    return round(float(np.clip(value, low * 0.4, high * 4)), 2)


def _random_timestamp(rng: np.random.Generator, odd_hour_bias: bool = False) -> datetime:
    """Draw a random timestamp within the trailing WINDOW_DAYS window."""
    days_ago = rng.uniform(0, WINDOW_DAYS)
    base = datetime.utcnow() - timedelta(days=days_ago)
    if odd_hour_bias and rng.random() < 0.7:
        hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
    else:
        hour = int(rng.integers(0, 24))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return base.replace(hour=hour, minute=minute, second=second, microsecond=0)


def _choose_channel(rng: np.random.Generator, category: str, force_online: bool = False) -> str:
    if force_online:
        return "online"
    online_prob = 0.75 if category in ONLINE_LEANING_CATEGORIES else 0.35
    return "online" if rng.random() < online_prob else "in_store"


def generate_categories() -> pd.DataFrame:
    return pd.DataFrame({"name": CATEGORIES})


def generate_customers(rng: np.random.Generator, faker: Faker) -> pd.DataFrame:
    rows = []
    for i in range(1, config.NUM_CUSTOMERS + 1):
        customer_id = f"CUST{i:05d}"
        name = faker.name()
        rows.append({
            "customer_id": customer_id,
            "name": name,
            "email": faker.unique.email(),
            "city": rng.choice(CITIES),
            "signup_date": faker.date_between(start_date="-3y", end_date="-30d").isoformat(),
        })
    return pd.DataFrame(rows)


def generate_merchants(rng: np.random.Generator, faker: Faker) -> pd.DataFrame:
    rows = []
    compromised_indices = set(
        rng.choice(config.NUM_MERCHANTS, size=config.NUM_COMPROMISED_MERCHANTS, replace=False)
    )
    for i in range(config.NUM_MERCHANTS):
        merchant_id = f"MERCH{i + 1:04d}"
        rows.append({
            "merchant_id": merchant_id,
            "name": faker.company(),
            "category": rng.choice(CATEGORIES),
            "city": rng.choice(CITIES),
            "is_compromised": i in compromised_indices,
        })
    return pd.DataFrame(rows)


def generate_devices(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(1, config.NUM_DEVICES + 1):
        rows.append({
            "device_id": f"DEV{i:06d}",
            "device_type": rng.choice(DEVICE_TYPES, p=DEVICE_TYPE_WEIGHTS),
        })
    return pd.DataFrame(rows)


def _assign_customer_devices(rng: np.random.Generator, customer_ids, own_device_pool) -> dict:
    """Give each customer 1-2 personal devices drawn from the non-ring pool."""
    assignment = {}
    pool = list(own_device_pool)
    for customer_id in customer_ids:
        n = 1 if rng.random() < 0.7 else 2
        devices = list(rng.choice(pool, size=min(n, len(pool)), replace=False))
        assignment[customer_id] = devices
    return assignment


def _build_ring_membership(rng: np.random.Generator, customer_ids, ring_device_pool) -> dict:
    """Assign each ring device a small group of unrelated customers who share it."""
    membership = {}
    for device_id in ring_device_pool:
        group_size = int(rng.integers(4, 9))
        members = list(rng.choice(customer_ids, size=group_size, replace=False))
        membership[device_id] = members
    return membership


def generate_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    customer_ids = customers["customer_id"].tolist()
    device_ids = devices["device_id"].tolist()
    merchants_by_category = {
        cat: merchants.loc[merchants["category"] == cat, "merchant_id"].tolist()
        for cat in CATEGORIES
    }
    merchant_category = dict(zip(merchants["merchant_id"], merchants["category"]))
    compromised_merchants = merchants.loc[merchants["is_compromised"], "merchant_id"].tolist()

    ring_devices = list(rng.choice(device_ids, size=config.NUM_FRAUD_RING_DEVICES, replace=False))
    own_device_pool = [d for d in device_ids if d not in ring_devices]
    ring_membership = _build_ring_membership(rng, customer_ids, ring_devices)
    customer_devices = _assign_customer_devices(rng, customer_ids, own_device_pool)

    # Each customer leans toward 1-3 favorite categories, mimicking real habits.
    customer_prefs = {
        cid: list(rng.choice(CATEGORIES, size=int(rng.integers(1, 4)), replace=False))
        for cid in customer_ids
    }

    rows: list[dict] = []

    # --- 1. Structural fraud: device-sharing rings -------------------------
    ring_txn_target = 0
    for device_id, members in ring_membership.items():
        burst = int(rng.integers(6, 21))
        ring_txn_target += burst
        for _ in range(burst):
            customer_id = rng.choice(members)
            merchant_id = (
                rng.choice(compromised_merchants) if rng.random() < 0.7 and compromised_merchants
                else rng.choice(merchants["merchant_id"].tolist())
            )
            category = merchant_category[merchant_id]
            rows.append({
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "device_id": device_id,
                "amount": _sample_amount(rng, category, elevated=True),
                "timestamp": _random_timestamp(rng, odd_hour_bias=True),
                "channel": _choose_channel(rng, category, force_online=True),
                "is_fraud": int(rng.random() < 0.85),
            })

    # --- 2. Structural fraud: compromised merchants -------------------------
    for merchant_id in compromised_merchants:
        category = merchant_category[merchant_id]
        extra = int(rng.integers(40, 91))
        for _ in range(extra):
            customer_id = rng.choice(customer_ids)
            device_id = rng.choice(customer_devices.get(customer_id, own_device_pool))
            rows.append({
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "device_id": device_id,
                "amount": _sample_amount(rng, category, elevated=True),
                "timestamp": _random_timestamp(rng, odd_hour_bias=True),
                "channel": _choose_channel(rng, category),
                "is_fraud": int(rng.random() < 0.6),
            })

    # --- 3. Organic baseline traffic ----------------------------------------
    remaining = max(config.NUM_TRANSACTIONS - len(rows), 0)
    for _ in range(remaining):
        customer_id = rng.choice(customer_ids)
        prefs = customer_prefs[customer_id]
        category = rng.choice(prefs) if rng.random() < 0.8 else rng.choice(CATEGORIES)
        candidates = merchants_by_category.get(category) or merchants["merchant_id"].tolist()
        merchant_id = rng.choice(candidates)
        device_id = rng.choice(customer_devices.get(customer_id, own_device_pool))
        odd_hour = rng.random() < 0.05
        amount = _sample_amount(rng, category, elevated=(rng.random() < 0.03))
        # A little unstructured fraud so the model learns generalizable signal,
        # not just "was this row part of an injected structure".
        base_prob = 0.004
        if odd_hour:
            base_prob += 0.01
        low, high = CATEGORY_AMOUNT_RANGE[category]
        if amount > high * 1.5:
            base_prob += 0.02
        rows.append({
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": amount,
            "timestamp": _random_timestamp(rng, odd_hour_bias=odd_hour),
            "channel": _choose_channel(rng, category),
            "is_fraud": int(rng.random() < base_prob),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{idx + 1:07d}" for idx in df.index])
    df["timestamp"] = df["timestamp"].apply(lambda dt: dt.isoformat())
    return df


def generate_all(seed: int = config.RANDOM_SEED) -> GeneratedDataset:
    """Generate the full synthetic dataset deterministically for a given seed."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    categories = generate_categories()
    customers = generate_customers(rng, faker)
    merchants = generate_merchants(rng, faker)
    devices = generate_devices(rng)
    transactions = generate_transactions(rng, customers, merchants, devices)

    return GeneratedDataset(
        customers=customers,
        merchants=merchants,
        categories=categories,
        devices=devices,
        transactions=transactions,
    )


def save_dataset(dataset: GeneratedDataset) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.customers.to_csv(config.CUSTOMERS_CSV, index=False)
    dataset.merchants.to_csv(config.MERCHANTS_CSV, index=False)
    dataset.categories.to_csv(config.CATEGORIES_CSV, index=False)
    dataset.devices.to_csv(config.DEVICES_CSV, index=False)
    dataset.transactions.to_csv(config.TRANSACTIONS_CSV, index=False)
