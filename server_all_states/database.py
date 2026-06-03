"""
Database client for the ISOLATED all_npi_states API.

Same Supabase project as the postie server, but targets the `all_npi_states` table
(synced from Firestore by etl/). Key differences from the postie SupabaseClient:
  - case-INSENSITIVE name matching (data is title-case, not UPPERCASE)
  - accepts 2-letter state codes (e.g. "CA") and maps them to the table's full
    lowercase names (e.g. "california")
  - dates come from start_date_iso / end_date_iso (timestamptz), not the raw strings
  - county / agency_type don't exist upstream -> county lookups return None,
    agency_type filtering is skipped (records default to POLICE in the model)
"""
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from supabase import create_client, Client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.src import PostEmploymentRecord, CandidateQuery

# 2-letter code -> table state value (lowercase, hyphenated, as stored in all_npi_states)
STATE_CODE_TO_NAME = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "columbia", "FL": "florida",
    "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana",
    "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine",
    "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new-hampshire",
    "NJ": "new-jersey", "NM": "new-mexico", "NY": "new-york", "NC": "north-carolina", "ND": "north-dakota",
    "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania", "RI": "rhode-island",
    "SC": "south-carolina", "SD": "south-dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west-virginia", "WI": "wisconsin",
    "WY": "wyoming",
}


def _naive_dt(v) -> Optional[datetime]:
    """Parse an ISO timestamp (possibly tz-aware, e.g. '...Z') to a tz-NAIVE datetime,
    matching postie's tz-naive contract so downstream (matching, features, Excel) is consistent."""
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _normalize_state(state: Optional[str]) -> Optional[str]:
    """Accept 'CA' or 'california' (any case) -> table value 'california'."""
    if not state:
        return None
    s = state.strip()
    if len(s) == 2 and s.upper() in STATE_CODE_TO_NAME:
        return STATE_CODE_TO_NAME[s.upper()]
    return s.lower()


class AllStatesClient:
    """Supabase operations over the all_npi_states table."""

    def __init__(self, url: str, key: str, table: str = "all_npi_states"):
        self.supabase: Client = create_client(url, key)
        self.table = table

    # ---- shared helpers ----------------------------------------------------
    def _apply_filters(self, query, q: CandidateQuery):
        if q.first_name:
            query = query.ilike("first_name", f"%{q.first_name}%")
        if q.last_name:
            query = query.ilike("last_name", f"%{q.last_name}%")
        if q.agency:
            query = query.ilike("agency_name", f"%{q.agency}%")
        if q.state:
            query = query.eq("state", _normalize_state(q.state))
        return query

    def _transform_record(self, r: Dict[str, Any]) -> PostEmploymentRecord:
        # dates: use the parsed ISO timestamps (raw start_date is a string like "4/24/00")
        return PostEmploymentRecord(
            post_person_nbr=r.get("person_nbr", "") or "",
            post_first_name=r.get("first_name", "") or "",
            post_middle_name=r.get("middle_name", "") or "",
            post_last_name=r.get("last_name", "") or "",
            post_suffix="",
            post_agency_name=r.get("agency_name", "") or "",
            post_agency_type="POLICE",                 # no agency_type upstream
            post_start_date=_naive_dt(r.get("start_date_iso")),
            post_end_date=_naive_dt(r.get("end_date_iso")),
            post_separation_reason=r.get("separation_reason", "") or "",
            state=r.get("state", "") or "",
            county="",                                  # no county upstream
        )

    # ---- endpoints ---------------------------------------------------------
    def get_post_employment_records(self, q: CandidateQuery) -> List[PostEmploymentRecord]:
        query = self.supabase.table(self.table).select("*")
        query = self._apply_filters(query, q)
        if q.offset and q.offset > 0:
            query = query.range(q.offset, q.offset + (q.limit or 1000) - 1)
        elif q.limit:
            query = query.limit(q.limit)
        resp = query.execute()
        return [self._transform_record(r) for r in (resp.data or [])]

    def get_post_employment_count(self, q: Optional[CandidateQuery] = None) -> int:
        query = self.supabase.table(self.table).select("person_nbr", count="exact")
        if q:
            query = self._apply_filters(query, q)
        return query.execute().count or 0

    def get_post_stats(self) -> Dict[str, Any]:
        total = self.supabase.table(self.table).select("person_nbr", count="exact").execute().count or 0
        return {"table": self.table, "total_records": total}

    def get_candidates_for_mention(self, q: CandidateQuery) -> List[PostEmploymentRecord]:
        """Two-query wide net (case-insensitive), merged + deduped. Mirrors the postie
        contract but skips county/agency_type filtering (not available upstream)."""
        state_val = _normalize_state(q.state)

        def base():
            qq = self.supabase.table(self.table).select("*")
            if state_val:
                qq = qq.eq("state", state_val)
            return qq

        # Query 1: first-name 2-char prefix (case-insensitive) + exact last name (case-insensitive)
        q1 = base()
        if q.first_name and len(q.first_name) >= 2:
            q1 = q1.ilike("first_name", f"{q.first_name[:2]}%")
        if q.last_name:
            q1 = q1.ilike("last_name", q.last_name)   # ilike w/o % == case-insensitive exact

        # Query 2: exact first name (case-insensitive) + last-name 2-char prefix
        q2 = base()
        if q.first_name:
            q2 = q2.ilike("first_name", q.first_name)
        if q.last_name and len(q.last_name) >= 2:
            q2 = q2.ilike("last_name", f"{q.last_name[:2]}%")

        rows = (q1.execute().data or []) + (q2.execute().data or [])
        seen, out = set(), []
        for r in rows:
            key = (r.get("person_nbr"), r.get("start_date"), r.get("end_date"),
                   r.get("agency_name"), r.get("state"))
            if key not in seen:
                seen.add(key)
                out.append(self._transform_record(r))
        return out

    def get_officers_by_name(self, first_name: str, last_name: str,
                             state: Optional[str] = None) -> List[PostEmploymentRecord]:
        if not first_name or not last_name:
            return []
        q = (self.supabase.table(self.table).select("*")
             .ilike("first_name", first_name)
             .ilike("last_name", f"{last_name}%"))
        if state:
            # state-scope the same-name lookup so ambiguity is judged within one state
            # (the postie table was CA-only, so this preserves the original semantics)
            q = q.eq("state", _normalize_state(state))
        return [self._transform_record(r) for r in (q.execute().data or [])]

    def get_batch_name_uniqueness(self, names: List[List[str]]) -> Dict[str, int]:
        """Count distinct person_nbr per (first,last) name. Case-insensitive exact match."""
        out: Dict[str, int] = {}
        for pair in names:
            first, last = pair[0], pair[1]
            resp = (self.supabase.table(self.table).select("person_nbr")
                    .ilike("first_name", first).ilike("last_name", last).execute())
            persons = {r.get("person_nbr") for r in (resp.data or [])}
            out[f"{first}|{last}"] = len(persons)
        return out

    # ---- county is not available upstream: explicit no-ops -----------------
    def get_county_for_agency(self, agency_name: str) -> Optional[str]:
        return None

    def get_counties_for_agencies_batch(self, agency_names: List[str]) -> Dict[str, None]:
        return {a: None for a in agency_names}

    def get_candidates_for_mentions_batch(self, mentions: List[Dict]) -> Dict[str, List[PostEmploymentRecord]]:
        results: Dict[str, List[PostEmploymentRecord]] = {}
        for i, m in enumerate(mentions):
            q = CandidateQuery(
                first_name=m.get("first_name"), last_name=m.get("last_name"),
                start_year=m.get("start_year"), end_year=m.get("end_year"),
                state=m.get("state"),
            )
            results[str(i)] = self.get_candidates_for_mention(q)
        return results
