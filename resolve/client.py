"""HTTP client for the NPI employment API.

Talks to whichever API `NPI_API_URL` points at (the all-states API on :8001 or the
legacy postie API on :8000) — the pipeline switches backends with no code change.
The requests session is injectable for testing.

Ported from the legacy resolve/src/api.py; imports the shared Pydantic models.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

from shared.models import PostEmploymentRecord

logger = logging.getLogger(__name__)


def _agency_type_value(agency_type) -> str:
    """Accept an AgencyType enum or a plain string."""
    return getattr(agency_type, "value", str(agency_type))


class NPIClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 180, session=None):
        base_url = base_url or os.environ.get("NPI_API_URL", "http://localhost:8000")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health_check(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/", timeout=self.timeout).status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_post_employment_records(
        self, limit=None, offset=0, first_name=None, last_name=None, agency=None, state=None
    ) -> List[PostEmploymentRecord]:
        try:
            params = {"offset": offset}
            for k, v in dict(limit=limit, first_name=first_name, last_name=last_name,
                             agency=agency, state=state).items():
                if v:
                    params[k] = v
            resp = self.session.get(f"{self.base_url}/post/employment", params=params,
                                    timeout=self.timeout)
            resp.raise_for_status()
            return [PostEmploymentRecord(**r) for r in resp.json()]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get POST employment records: {e}")
            return []

    def get_candidates_for_mention(
        self, first_name: str, last_name: str, incident_year: int = 2018,
        agency_type="POLICE", state: Optional[str] = None,
    ) -> List[PostEmploymentRecord]:
        try:
            params = {
                "first_name": first_name,
                "last_name": last_name,
                "agency_type": _agency_type_value(agency_type),
                "start_year": incident_year,
                "end_year": incident_year,
            }
            if state:
                params["state"] = state
            resp = self.session.get(f"{self.base_url}/post/candidates", params=params,
                                    timeout=self.timeout)
            resp.raise_for_status()
            return [PostEmploymentRecord(**r) for r in resp.json()]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get candidates: {e}")
            return []

    def get_officers_by_name(
        self, first_name: str, last_name: str, state: Optional[str] = None
    ) -> List[PostEmploymentRecord]:
        try:
            params = {"first_name": first_name, "last_name": last_name}
            if state:
                params["state"] = state
            resp = self.session.get(f"{self.base_url}/post/officers/by-name", params=params,
                                    timeout=self.timeout)
            resp.raise_for_status()
            return [PostEmploymentRecord(**r) for r in resp.json()]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get officers by name: {e}")
            return []

    def get_county_for_agency(self, agency_name: str) -> Optional[str]:
        try:
            resp = self.session.get(f"{self.base_url}/post/agency/county",
                                    params={"agency_name": agency_name}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("county")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get county for agency: {e}")
            return None

    def get_post_employment_count(self, first_name=None, last_name=None, agency=None,
                                  state=None) -> Optional[int]:
        try:
            params = {k: v for k, v in dict(first_name=first_name, last_name=last_name,
                                            agency=agency, state=state).items() if v}
            resp = self.session.get(f"{self.base_url}/post/employment/count", params=params,
                                    timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("total_records")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get POST employment count: {e}")
            return None
