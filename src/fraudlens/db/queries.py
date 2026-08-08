"""All Cypher queries used by Fraudlens, in one place.

Every query is parameterized (no string-concatenated Cypher) and returns
plain Python dict/list structures so callers never touch the driver
directly. Query names map to the three graph-centric questions the
assignment asks the UI to answer:

  Q1 merchants_connected_to_fraud   -- what merchants are tied to fraud?
  Q2 customer_network               -- what's connected to this customer? (multi-hop)
  Q3 merchant_connected_customers   -- what other customers touch a suspicious merchant?

plus shared_device_rings, which is the query a relational schema handles
awkwardly: finding devices reused across otherwise-unrelated customers
requires a self-join and group-having in SQL, and grows worse the moment you
want to expand outward from the ring (ring_expansion) -- in Cypher both are
a direct pattern match.
"""
from __future__ import annotations

from fraudlens.db.connection import CognoDBConnection

# --- Dashboard -------------------------------------------------------------

_DASHBOARD_STATS = """
MATCH (t:Transaction)
RETURN
  count(t) AS total_transactions,
  sum(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
  sum(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END) AS fraud_amount,
  sum(t.amount) AS total_amount,
  avg(t.fraud_probability) AS avg_fraud_probability,
  sum(CASE WHEN t.fraud_probability >= 0.5 THEN 1 ELSE 0 END) AS high_risk_count
"""

_ENTITY_COUNTS = """
MATCH (c:Customer)
WITH count(c) AS customers
MATCH (m:Merchant)
WITH customers, count(m) AS merchants
MATCH (d:Device)
RETURN customers, merchants, count(d) AS devices
"""

_FRAUD_BY_CATEGORY = """
MATCH (t:Transaction)-[:AT_MERCHANT]->(m:Merchant)-[:IN_CATEGORY]->(cat:Category)
WHERE t.is_fraud = 1
RETURN cat.name AS category, count(t) AS fraud_count, sum(t.amount) AS fraud_amount
ORDER BY fraud_count DESC
"""

_RECENT_HIGH_RISK = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)-[:AT_MERCHANT]->(m:Merchant)
WHERE t.fraud_probability IS NOT NULL
RETURN t.transaction_id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp,
       t.fraud_probability AS fraud_probability, t.is_fraud AS is_fraud,
       c.customer_id AS customer_id, c.name AS customer_name,
       m.merchant_id AS merchant_id, m.name AS merchant_name
ORDER BY t.fraud_probability DESC
LIMIT $limit
"""


def dashboard_stats(conn: CognoDBConnection) -> dict:
    stats = conn.run_query(_DASHBOARD_STATS)[0]
    counts = conn.run_query(_ENTITY_COUNTS)[0]
    return {**stats, **counts}


def fraud_by_category(conn: CognoDBConnection) -> list[dict]:
    return conn.run_query(_FRAUD_BY_CATEGORY)


def recent_high_risk_transactions(conn: CognoDBConnection, limit: int = 10) -> list[dict]:
    return conn.run_query(_RECENT_HIGH_RISK, {"limit": limit})


# --- Q1: merchants connected to fraudulent activity -------------------------

_MERCHANTS_CONNECTED_TO_FRAUD = """
MATCH (m:Merchant)<-[:AT_MERCHANT]-(t:Transaction)
WHERE t.is_fraud = 1
WITH m, count(t) AS fraud_txn_count, sum(t.amount) AS fraud_amount
MATCH (m)<-[:AT_MERCHANT]-(:Transaction)<-[:MADE]-(c:Customer)
RETURN m.merchant_id AS merchant_id, m.name AS name, m.city AS city,
       m.is_compromised AS is_compromised, fraud_txn_count, fraud_amount,
       count(DISTINCT c) AS connected_customers
ORDER BY fraud_txn_count DESC
LIMIT $limit
"""


def merchants_connected_to_fraud(conn: CognoDBConnection, limit: int = 20) -> list[dict]:
    return conn.run_query(_MERCHANTS_CONNECTED_TO_FRAUD, {"limit": limit})


# --- Q2: everything connected to a customer (multi-hop) ---------------------

_CUSTOMER_NETWORK = """
MATCH (c:Customer {customer_id: $customer_id})-[:MADE]->(t:Transaction)
      -[:AT_MERCHANT]->(m:Merchant)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)
RETURN c.customer_id AS customer_id, c.name AS customer_name, c.city AS customer_city,
       t.transaction_id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp,
       t.is_fraud AS is_fraud, t.fraud_probability AS fraud_probability, t.channel AS channel,
       m.merchant_id AS merchant_id, m.name AS merchant_name, m.is_compromised AS merchant_compromised,
       cat.name AS category, d.device_id AS device_id
ORDER BY t.timestamp DESC
LIMIT $limit
"""


def customer_network(conn: CognoDBConnection, customer_id: str, limit: int = 200) -> list[dict]:
    """3-hop traversal: Customer -> Transaction -> Merchant -> Category."""
    return conn.run_query(_CUSTOMER_NETWORK, {"customer_id": customer_id, "limit": limit})


# --- Q3: other customers connected to a suspicious merchant -----------------

_MERCHANT_CONNECTED_CUSTOMERS = """
MATCH (m:Merchant {merchant_id: $merchant_id})<-[:AT_MERCHANT]-(t:Transaction)<-[:MADE]-(c:Customer)
WHERE t.is_fraud = 1 OR t.fraud_probability >= $risk_threshold
RETURN c.customer_id AS customer_id, c.name AS name, c.city AS city,
       count(t) AS suspicious_txn_count, sum(t.amount) AS suspicious_amount
ORDER BY suspicious_txn_count DESC
LIMIT $limit
"""


def merchant_connected_customers(
    conn: CognoDBConnection, merchant_id: str, risk_threshold: float = 0.5, limit: int = 50
) -> list[dict]:
    return conn.run_query(
        _MERCHANT_CONNECTED_CUSTOMERS,
        {"merchant_id": merchant_id, "risk_threshold": risk_threshold, "limit": limit},
    )


# --- Shared-device fraud rings (the relational-awkward query) ---------------

_SHARED_DEVICE_RINGS = """
MATCH (d:Device)<-[:USED_DEVICE]-(t:Transaction)<-[:MADE]-(c:Customer)
WITH d, collect(DISTINCT c.customer_id) AS customer_ids, count(DISTINCT c) AS customer_count,
     sum(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_txn_count
WHERE customer_count > 1 AND fraud_txn_count > 0
RETURN d.device_id AS device_id, customer_count, fraud_txn_count, customer_ids
ORDER BY fraud_txn_count DESC, customer_count DESC
LIMIT $limit
"""

_RING_EXPANSION = """
MATCH (d:Device {device_id: $device_id})<-[:USED_DEVICE]-(:Transaction)<-[:MADE]-(c:Customer)
WITH DISTINCT c
MATCH (c)-[:MADE]->(:Transaction)-[:USED_DEVICE]->(other:Device)
WHERE other.device_id <> $device_id
RETURN c.customer_id AS customer_id, c.name AS name,
       collect(DISTINCT other.device_id) AS other_devices
"""


def shared_device_rings(conn: CognoDBConnection, limit: int = 15) -> list[dict]:
    """Devices shared by multiple customers with at least one fraudulent transaction.

    This is the query a relational schema struggles with: finding groups of
    otherwise-unrelated customers who share an entity (here, a device) needs
    a self-join + GROUP BY/HAVING in SQL, and following the ring one hop
    further (see ring_expansion) needs another hand-written join for every
    extra hop. In Cypher both are a direct pattern match.
    """
    return conn.run_query(_SHARED_DEVICE_RINGS, {"limit": limit})


def ring_expansion(conn: CognoDBConnection, device_id: str) -> list[dict]:
    """What other devices do the customers in this ring also use? (3-hop)."""
    return conn.run_query(_RING_EXPANSION, {"device_id": device_id})


# --- Browsing / search --------------------------------------------------

_LIST_CUSTOMERS = """
MATCH (c:Customer)
OPTIONAL MATCH (c)-[:MADE]->(t:Transaction)
WITH c, count(t) AS txn_count,
     sum(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
     avg(t.fraud_probability) AS avg_risk
WHERE $search IS NULL OR toLower(c.name) CONTAINS toLower($search)
      OR toLower(c.customer_id) CONTAINS toLower($search)
RETURN c.customer_id AS customer_id, c.name AS name, c.city AS city,
       txn_count, fraud_count, avg_risk
ORDER BY fraud_count DESC, avg_risk DESC
LIMIT $limit
"""

_LIST_MERCHANTS = """
MATCH (m:Merchant)
OPTIONAL MATCH (m)<-[:AT_MERCHANT]-(t:Transaction)
WITH m, count(t) AS txn_count, sum(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count
WHERE $search IS NULL OR toLower(m.name) CONTAINS toLower($search)
      OR toLower(m.merchant_id) CONTAINS toLower($search)
RETURN m.merchant_id AS merchant_id, m.name AS name, m.city AS city,
       m.is_compromised AS is_compromised, txn_count, fraud_count
ORDER BY fraud_count DESC
LIMIT $limit
"""

_TRANSACTION_DETAIL = """
MATCH (c:Customer)-[:MADE]->(t:Transaction {transaction_id: $transaction_id})
      -[:AT_MERCHANT]->(m:Merchant)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)
OPTIONAL MATCH (d)<-[:USED_DEVICE]-(:Transaction)<-[:MADE]-(other:Customer)
WHERE other.customer_id <> c.customer_id
RETURN c.customer_id AS customer_id, c.name AS customer_name, c.city AS customer_city,
       t.transaction_id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp,
       t.channel AS channel, t.is_fraud AS is_fraud, t.fraud_probability AS fraud_probability,
       m.merchant_id AS merchant_id, m.name AS merchant_name, m.city AS merchant_city,
       m.is_compromised AS merchant_compromised, cat.name AS category,
       d.device_id AS device_id, d.device_type AS device_type,
       count(DISTINCT other) AS device_shared_with_customers
"""

_SEARCH_TRANSACTIONS = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)-[:AT_MERCHANT]->(m:Merchant)
WHERE ($min_amount IS NULL OR t.amount >= $min_amount)
  AND ($fraud_only = false OR t.is_fraud = 1)
  AND ($min_risk IS NULL OR t.fraud_probability >= $min_risk)
  AND ($search IS NULL OR toLower(t.transaction_id) CONTAINS toLower($search)
       OR toLower(c.name) CONTAINS toLower($search)
       OR toLower(m.name) CONTAINS toLower($search))
RETURN t.transaction_id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp,
       t.is_fraud AS is_fraud, t.fraud_probability AS fraud_probability, t.channel AS channel,
       c.customer_id AS customer_id, c.name AS customer_name,
       m.merchant_id AS merchant_id, m.name AS merchant_name
ORDER BY t.timestamp DESC
LIMIT $limit
"""


def list_customers(conn: CognoDBConnection, search: str | None = None, limit: int = 50) -> list[dict]:
    return conn.run_query(_LIST_CUSTOMERS, {"search": search, "limit": limit})


def list_merchants(conn: CognoDBConnection, search: str | None = None, limit: int = 50) -> list[dict]:
    return conn.run_query(_LIST_MERCHANTS, {"search": search, "limit": limit})


def transaction_detail(conn: CognoDBConnection, transaction_id: str) -> dict | None:
    rows = conn.run_query(_TRANSACTION_DETAIL, {"transaction_id": transaction_id})
    return rows[0] if rows else None


def search_transactions(
    conn: CognoDBConnection,
    search: str | None = None,
    min_amount: float | None = None,
    fraud_only: bool = False,
    min_risk: float | None = None,
    limit: int = 100,
) -> list[dict]:
    return conn.run_query(
        _SEARCH_TRANSACTIONS,
        {
            "search": search,
            "min_amount": min_amount,
            "fraud_only": fraud_only,
            "min_risk": min_risk,
            "limit": limit,
        },
    )


# --- Feature extraction for scoring (see scripts/score_transactions.py) -----

_RAW_TRANSACTIONS = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)-[:AT_MERCHANT]->(m:Merchant)
OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)
RETURN t.transaction_id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp,
       t.channel AS channel, t.is_fraud AS is_fraud,
       c.customer_id AS customer_id, m.merchant_id AS merchant_id, d.device_id AS device_id
SKIP $skip LIMIT $limit
"""

_CUSTOMER_AMOUNT_STATS = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)
RETURN c.customer_id AS customer_id, avg(t.amount) AS mean_amount, stDev(t.amount) AS std_amount
"""

_MERCHANT_FRAUD_RATES = """
MATCH (m:Merchant)<-[:AT_MERCHANT]-(t:Transaction)
RETURN m.merchant_id AS merchant_id,
       avg(CASE WHEN t.is_fraud = 1 THEN 1.0 ELSE 0.0 END) AS fraud_rate
"""

_DEVICE_SHARED_COUNTS = """
MATCH (d:Device)<-[:USED_DEVICE]-(:Transaction)<-[:MADE]-(c:Customer)
RETURN d.device_id AS device_id, count(DISTINCT c) AS shared_customer_count
"""

_FIRST_MERCHANT_VISIT = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)-[:AT_MERCHANT]->(m:Merchant)
RETURN c.customer_id AS customer_id, m.merchant_id AS merchant_id, min(t.timestamp) AS first_seen
"""


def count_transactions(conn: CognoDBConnection) -> int:
    return conn.run_query("MATCH (t:Transaction) RETURN count(t) AS n")[0]["n"]


def fetch_raw_transactions_batch(conn: CognoDBConnection, skip: int, limit: int) -> list[dict]:
    return conn.run_query(_RAW_TRANSACTIONS, {"skip": skip, "limit": limit})


def fetch_customer_amount_stats(conn: CognoDBConnection) -> list[dict]:
    return conn.run_query(_CUSTOMER_AMOUNT_STATS)


def fetch_merchant_fraud_rates(conn: CognoDBConnection) -> list[dict]:
    return conn.run_query(_MERCHANT_FRAUD_RATES)


def fetch_device_shared_counts(conn: CognoDBConnection) -> list[dict]:
    return conn.run_query(_DEVICE_SHARED_COUNTS)


def fetch_first_merchant_visit(conn: CognoDBConnection) -> list[dict]:
    return conn.run_query(_FIRST_MERCHANT_VISIT)


# --- Writing scores back --------------------------------------------------

_UPDATE_FRAUD_PROBABILITIES = """
UNWIND $rows AS row
MATCH (t:Transaction {transaction_id: row.transaction_id})
SET t.fraud_probability = row.fraud_probability
"""

_UPDATE_CUSTOMER_RISK_SCORES = """
MATCH (c:Customer)-[:MADE]->(t:Transaction)
WHERE t.fraud_probability IS NOT NULL
WITH c, avg(t.fraud_probability) AS avg_risk
SET c.risk_score = avg_risk
"""


def update_fraud_probabilities(conn: CognoDBConnection, rows: list[dict]) -> dict:
    """Batch-write {transaction_id, fraud_probability} rows via UNWIND."""
    return conn.run_write_batch(_UPDATE_FRAUD_PROBABILITIES, rows)


def update_customer_risk_scores(conn: CognoDBConnection) -> dict:
    return conn.run_write(_UPDATE_CUSTOMER_RISK_SCORES)
