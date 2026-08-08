"""CognoDB connection handling.

CognoDB speaks openCypher over Bolt and is wire-compatible with the official
Neo4j Python driver, so no custom SDK is involved here -- just the standard
`neo4j` package pointed at the CognoDB URI.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from fraudlens.config import CognoDBSettings, get_cognodb_settings

_COUNTER_FIELDS = [
    "nodes_created", "nodes_deleted", "relationships_created", "relationships_deleted",
    "properties_set", "labels_added", "labels_removed", "indexes_added", "indexes_removed",
    "constraints_added", "constraints_removed",
]


def _counters_to_dict(counters: Any) -> dict:
    """SummaryCounters isn't dict()-able; pull the fields we care about explicitly."""
    return {field: getattr(counters, field, 0) for field in _COUNTER_FIELDS}


class DatabaseUnavailableError(RuntimeError):
    """Raised when CognoDB cannot be reached or the query fails."""


def build_driver(settings: CognoDBSettings) -> Driver:
    """Create a Neo4j driver configured for CognoDB, verifying connectivity."""
    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    driver.verify_connectivity()
    return driver


class CognoDBConnection:
    """Thin wrapper around a Neo4j driver with lazy connect and clear errors.

    Kept intentionally small: one driver, one place that translates driver
    exceptions into DatabaseUnavailableError so callers (Streamlit pages,
    scripts) only need to handle one exception type.
    """

    def __init__(self, settings: CognoDBSettings | None = None) -> None:
        self._settings = settings or get_cognodb_settings()
        self._driver: Driver | None = None

    def connect(self) -> None:
        if not self._settings.is_configured:
            raise DatabaseUnavailableError("CognoDB connection is not configured.")
        try:
            self._driver = build_driver(self._settings)
        except (ServiceUnavailable, Neo4jError, ValueError) as exc:
            raise DatabaseUnavailableError(f"Could not connect to CognoDB: {exc}") from exc

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "CognoDBConnection":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def run_query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Run a single parameterized Cypher query and return records as dicts."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except (ServiceUnavailable, Neo4jError) as exc:
            raise DatabaseUnavailableError(f"CognoDB query failed: {exc}") from exc

    def run_write(self, cypher: str, parameters: dict | None = None) -> dict:
        """Run a single parameterized write Cypher query and return summary counters."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters or {})
                summary = result.consume()
                return _counters_to_dict(summary.counters)
        except (ServiceUnavailable, Neo4jError) as exc:
            raise DatabaseUnavailableError(f"CognoDB write failed: {exc}") from exc

    def run_write_batch(self, cypher: str, rows: list[dict], batch_param_name: str = "rows") -> dict:
        """Run a single UNWIND-based batch write. `cypher` must reference $<batch_param_name>."""
        return self.run_write(cypher, {batch_param_name: rows})


@contextmanager
def get_connection(settings: CognoDBSettings | None = None) -> Iterator[CognoDBConnection]:
    """Context-managed CognoDB connection for scripts."""
    conn = CognoDBConnection(settings)
    try:
        conn.connect()
        yield conn
    finally:
        conn.close()
