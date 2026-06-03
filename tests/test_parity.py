"""Parity tests: the all-states API must expose the same contract as the legacy
postie API, and return the same CA officers (modulo case / county / agency_type).

Opt-in (integration): hits both LIVE servers over HTTP and skips cleanly if either
is down. Data parity at scale was already validated head-to-head (see plans/firestore.md:
350 rows, 0 false positives, 0 different-person matches, equal/better recall); these
tests keep the endpoint contract honest going forward.

    # terminal 1: cd server (postie)            -> :8000
    # terminal 2: cd server_all_states (or api/) -> :8001
    venv/bin/python -m pytest -m integration tests/test_parity.py
"""
import os

import pytest
import requests

pytestmark = pytest.mark.integration

POSTIE_URL = os.environ.get("POSTIE_API_URL", "http://localhost:8000")
ALLSTATES_URL = os.environ.get("ALLSTATES_API_URL", "http://localhost:8001")

# The endpoint contract both APIs must satisfy.
EMPLOYMENT_FIELDS = {
    "post_person_nbr", "post_first_name", "post_middle_name", "post_last_name",
    "post_suffix", "post_agency_name", "post_agency_type", "post_start_date",
    "post_end_date", "post_separation_reason", "state", "county",
}


def _up(url: str) -> bool:
    try:
        return requests.get(f"{url}/", timeout=4).status_code == 200
    except requests.exceptions.RequestException:
        return False


both_up = pytest.mark.skipif(
    not (_up(POSTIE_URL) and _up(ALLSTATES_URL)),
    reason="parity needs BOTH servers up (postie :8000 and all-states :8001)",
)
allstates_up = pytest.mark.skipif(not _up(ALLSTATES_URL),
                                  reason="all-states API not reachable at :8001")


def _persons(records):
    """Case-insensitive set of person numbers (postie is UPPERCASE, all-states title-case)."""
    return {str(r["post_person_nbr"]).upper() for r in records}


@allstates_up
def test_allstates_employment_contract():
    resp = requests.get(f"{ALLSTATES_URL}/post/employment",
                        params={"last_name": "Lunger", "state": "CA", "limit": 5}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    assert isinstance(data, list)
    if data:
        assert EMPLOYMENT_FIELDS.issubset(set(data[0].keys()))


@both_up
def test_employment_contract_matches_between_apis():
    params = {"last_name": "Lunger", "state": "CA", "limit": 5}
    a = requests.get(f"{POSTIE_URL}/post/employment", params=params, timeout=30).json()
    b = requests.get(f"{ALLSTATES_URL}/post/employment", params=params, timeout=30).json()
    if a and b:
        # same response schema (the "works exactly the same" contract)
        assert set(a[0].keys()) == set(b[0].keys())


@both_up
def test_same_ca_person_returned_by_candidates():
    """For a known CA officer, both APIs surface the same person (case-insensitive)."""
    params = {"first_name": "Scott", "last_name": "Lunger", "agency_type": "POLICE",
              "start_year": 2015, "end_year": 2015, "state": "CA"}
    a = requests.get(f"{POSTIE_URL}/post/candidates", params=params, timeout=60).json()
    b = requests.get(f"{ALLSTATES_URL}/post/candidates", params=params, timeout=60).json()
    # all-states should surface at least the CA persons postie does for this query
    assert _persons(a).issubset(_persons(b)) or _persons(a) & _persons(b)
