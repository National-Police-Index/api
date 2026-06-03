"""Unit tests for the NPI HTTP client (no network — a fake session is injected)."""
import os

from shared.models import PostEmploymentRecord
from resolve.client import NPIClient


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records the last GET and returns a queued payload."""
    def __init__(self, payload):
        self._payload = payload
        self.last_url = None
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.last_url = url
        self.last_params = params or {}
        return FakeResponse(self._payload)


_REC = {
    "post_person_nbr": "P1", "post_first_name": "Scott", "post_last_name": "Lunger",
    "post_agency_name": "Hayward Police Department", "post_agency_type": "POLICE",
    "post_start_date": "2010-01-01T00:00:00", "post_end_date": "2020-01-01T00:00:00",
    "state": "CA", "county": None,
}


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("NPI_API_URL", "http://localhost:8001")
    assert NPIClient().base_url == "http://localhost:8001"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("NPI_API_URL", "http://localhost:8001")
    assert NPIClient(base_url="http://example/").base_url == "http://example"


def test_get_candidates_builds_params_and_parses():
    session = FakeSession([_REC])
    client = NPIClient(base_url="http://x", session=session)
    out = client.get_candidates_for_mention(
        first_name="Scott", last_name="Lunger", incident_year=2015, state="CA",
    )
    assert session.last_url.endswith("/post/candidates")
    assert session.last_params["first_name"] == "Scott"
    assert session.last_params["start_year"] == 2015
    assert session.last_params["state"] == "CA"
    assert len(out) == 1 and isinstance(out[0], PostEmploymentRecord)
    assert out[0].post_person_nbr == "P1"


def test_get_officers_by_name_includes_state_when_given():
    session = FakeSession([_REC])
    client = NPIClient(base_url="http://x", session=session)
    client.get_officers_by_name("Scott", "Lunger", state="CA")
    assert session.last_url.endswith("/post/officers/by-name")
    assert session.last_params["state"] == "CA"


def test_get_county_for_agency_parses_payload():
    session = FakeSession({"agency_name": "Hayward PD", "county": "Alameda"})
    client = NPIClient(base_url="http://x", session=session)
    assert client.get_county_for_agency("Hayward PD") == "Alameda"
