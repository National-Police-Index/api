"""End-to-end integration tests (opt-in): real API + real model + real LLM.

Run with the all-states API up on NPI_API_URL (default :8001):
    cd server_all_states && python3 src.py          # or api/ after the rename
    venv/bin/python -m pytest -m integration tests/test_integration_resolve.py

Skipped automatically when the API isn't reachable so the default suite stays green.
"""
import datetime
import os

import pytest

from resolve import PostMatcher, build_mention
from resolve.client import NPIClient

pytestmark = pytest.mark.integration

API_URL = os.environ.get("NPI_API_URL", "http://localhost:8001")


def _api_up() -> bool:
    try:
        return NPIClient(base_url=API_URL).health_check()
    except Exception:
        return False


requires_api = pytest.mark.skipif(not _api_up(), reason=f"API not reachable at {API_URL}")


@requires_api
def test_known_example_auto_matches_scott_lunger():
    """README ground truth: Scott Lunger / Hayward PD / 2015 / CA -> POST b04-j30."""
    matcher = PostMatcher(api_url=API_URL)
    mention = build_mention({
        "first_name": "Scott", "last_name": "Lunger", "source_agency": "Hayward Police Department",
        "incident_year": 2015, "state": "CA",
    })
    result = matcher.resolve_one(mention)
    assert result.status == "auto_matched"
    assert result.match["post_person_nbr"].lower().startswith("b04-j30")


@requires_api
def test_da_source_without_le_is_routed_to_review():
    """Non-LE guard: a DA source with no LE agency must not auto-match an LE record."""
    matcher = PostMatcher(api_url=API_URL)
    mention = build_mention({
        "first_name": "Scott", "last_name": "Lunger",
        "source_agency": "Alameda County District Attorney",
        "incident_year": 2015, "state": "CA",
    })
    result = matcher.resolve_one(mention)
    assert result.status == "review"
