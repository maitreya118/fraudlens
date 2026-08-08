"""Interactive network visualizations (pyvis) for the Fraud Network page."""
from __future__ import annotations

from pyvis.network import Network

from fraudlens.ui.theme import NEUTRAL, RISK_HIGH, RISK_MEDIUM

_CUSTOMER_COLOR = "#4C6EF5"
_MERCHANT_COLOR = "#495057"
_CATEGORY_COLOR = "#adb5bd"
_DEVICE_COLOR = "#9c36b5"


def _to_html(net: Network) -> str:
    """Render a pyvis Network to a standalone HTML string.

    Uses generate_html() (renders the Jinja template in memory) rather than
    write_html(), which writes through a plain open()/cp1252 codec on
    Windows and breaks on the unicode glyphs used in node tooltips.
    """
    return net.generate_html(notebook=False)


def _transaction_color(is_fraud: int | None, fraud_probability: float | None) -> str:
    if is_fraud == 1:
        return RISK_HIGH
    if fraud_probability is not None and fraud_probability >= 0.5:
        return RISK_MEDIUM
    return NEUTRAL


def render_customer_network(rows: list[dict]) -> str | None:
    """Customer -> Transaction -> Merchant -> Category, as an interactive graph."""
    if not rows:
        return None

    net = Network(height="560px", width="100%", directed=True, cdn_resources="in_line")
    net.barnes_hut(spring_length=110)

    customer_id = rows[0]["customer_id"]
    net.add_node(
        customer_id, label=rows[0]["customer_name"], color=_CUSTOMER_COLOR,
        shape="star", size=32, title=f"Customer: {rows[0]['customer_name']}",
    )

    seen_merchants: set[str] = set()
    seen_categories: set[str] = set()

    for row in rows:
        txn_id = row["transaction_id"]
        color = _transaction_color(row.get("is_fraud"), row.get("fraud_probability"))
        risk = row.get("fraud_probability")
        risk_txt = f", risk {risk:.0%}" if risk is not None else ""
        net.add_node(
            txn_id, label=f"${row['amount']:.0f}", color=color, shape="dot", size=13,
            title=f"{txn_id} · ${row['amount']:.2f} · {row['timestamp']}{risk_txt}",
        )
        net.add_edge(customer_id, txn_id)

        merchant_id = row["merchant_id"]
        if merchant_id not in seen_merchants:
            merchant_color = RISK_HIGH if row.get("merchant_compromised") else _MERCHANT_COLOR
            net.add_node(
                merchant_id, label=row["merchant_name"], color=merchant_color,
                shape="square", size=20, title=f"Merchant: {row['merchant_name']}",
            )
            seen_merchants.add(merchant_id)
        net.add_edge(txn_id, merchant_id)

        category = row["category"]
        cat_node = f"cat::{category}"
        if cat_node not in seen_categories:
            net.add_node(cat_node, label=category, color=_CATEGORY_COLOR, shape="triangle", size=16)
            seen_categories.add(cat_node)
        net.add_edge(merchant_id, cat_node)

    return _to_html(net)


def render_ring_graph(device_id: str, customers: list[dict], expansion_rows: list[dict]) -> str | None:
    """Shared-device ring: the device at the center, ring members, and their other devices."""
    if not customers:
        return None

    net = Network(height="480px", width="100%", directed=False, cdn_resources="in_line")
    net.barnes_hut(spring_length=120)

    net.add_node(device_id, label=f"Device\n{device_id}", color=RISK_HIGH, shape="square", size=28)

    for customer in customers:
        cid = customer["customer_id"]
        net.add_node(cid, label=customer.get("name", cid), color=_CUSTOMER_COLOR, shape="dot", size=18)
        net.add_edge(device_id, cid)

    for row in expansion_rows:
        cid = row["customer_id"]
        for other_device in row.get("other_devices", []):
            net.add_node(other_device, label=other_device, color=_DEVICE_COLOR, shape="square", size=14)
            net.add_edge(cid, other_device)

    return _to_html(net)
