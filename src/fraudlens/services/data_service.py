"""Factory that selects the data service the UI should use.

Two-tier fallback, in order of preference:
  1. CognoDB configured and reachable  -> CognoDBService (live graph queries)
  2. CognoDB not configured             -> DemoDataService (local dataset)

If CognoDB *is* configured but unreachable, the connection error is left to
propagate rather than silently degrading to demo data -- the app should
show a clear error, not fake graph results.
"""
from __future__ import annotations

from fraudlens.config import is_demo_mode
from fraudlens.services.cognodb_service import CognoDBService
from fraudlens.services.demo_service import DemoDataService

DataService = CognoDBService | DemoDataService


def get_data_service() -> DataService:
    if is_demo_mode():
        return DemoDataService()
    return CognoDBService()
