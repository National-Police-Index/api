"""Candidate filtering for entity resolution.

Pure DataFrame transforms over POST employment records (as returned by the API),
factored out of the legacy generate_candidates(). The optional CA-rich filters
(county, agency_type) are applied only when the data supports them, so the same
code serves both the rich CA `postie` data and the lean all-states data.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

PREFIX_LEN = 2


def _to_naive(series: pd.Series) -> pd.Series:
    """Parse a date column (raw strings, tz-naive, or tz-aware ISO like '...Z')
    into tz-naive datetimes. Empty strings -> NaT."""
    s = series.replace("", pd.NaT)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def in_date_range(post: pd.DataFrame, incident_year: int, buffer: int = 1) -> pd.Series:
    """Boolean mask: employment overlaps [incident_year - buffer, incident_year + buffer].

    End dates that are empty or implausible (before 1950) are treated as "current"
    (filled with today), matching the legacy date-handling edge cases.
    """
    start_dates = _to_naive(post["post_start_date"])

    end_dates = _to_naive(post["post_end_date"])
    end_dates = end_dates.where((end_dates.isna()) | (end_dates.dt.year >= 1950), pd.NaT)
    end_dates = end_dates.fillna(pd.Timestamp.today())

    return (start_dates.dt.year <= incident_year + buffer) & (
        end_dates.dt.year >= incident_year - buffer
    )


def filter_by_name(
    post: pd.DataFrame, first_name: str, last_name: str, prefix_len: int = PREFIX_LEN
) -> pd.DataFrame:
    """Wide name net (case-insensitive): (first 2-char prefix + exact last) OR
    (exact first + last 2-char prefix). Catches nickname/data-entry variation."""
    fn_prefix = first_name[:prefix_len].casefold() if first_name else "z"

    fn_cand = post["post_first_name"].str[:prefix_len].str.casefold() == fn_prefix
    fn_full = post["post_first_name"].str.casefold() == (first_name or "").casefold()
    ln_cand = (
        post["post_last_name"].str[:prefix_len].str.casefold()
        == (last_name or "")[:prefix_len].casefold()
    )
    ln_full = post["post_last_name"].str.casefold() == (last_name or "").casefold()

    return pd.concat(
        [post.loc[fn_cand & ln_full], post.loc[fn_full & ln_cand]]
    ).drop_duplicates()


def filter_by_county(post: pd.DataFrame, source_county: str) -> pd.DataFrame:
    """Keep only persons with ANY employment record in `source_county`.

    Grouped by person so a single in-county stint keeps all of that person's rows.
    """
    has_match = post.groupby("post_person_nbr")["county"].apply(
        lambda counties: source_county in counties.values
    )
    valid = has_match[has_match].index
    return post[post["post_person_nbr"].isin(valid)]


def has_real_agency_type(post: pd.DataFrame) -> bool:
    """True if the data carries meaningful agency_type info (not uniformly the
    default 'POLICE'). All-states data is uniform POLICE -> False (skip the mask);
    CA `postie` data mixes POLICE/CORRECTIONS -> True (apply the mask)."""
    if "post_agency_type" not in post.columns or len(post) == 0:
        return False
    types = post["post_agency_type"].astype(str).str.upper()
    return bool((types != "POLICE").any())


def filter_by_agency_type(post: pd.DataFrame, agency_type: str) -> pd.DataFrame:
    """Keep records whose agency_type matches the mention's (case-insensitive)."""
    return post[post["post_agency_type"].astype(str).str.lower() == str(agency_type).lower()]


def select_candidates(
    post: pd.DataFrame,
    first_name: str,
    last_name: str,
    incident_year: int,
    agency_type: str = "POLICE",
    source_county: Optional[str] = None,
) -> pd.DataFrame:
    """Apply the full candidate filter chain, with county/agency_type applied only
    when the data supports them.

    Order matches the legacy pipeline: county (optional) -> agency_type (optional)
    -> date range -> name net.
    """
    if len(post) == 0:
        return post

    is_corrections = str(agency_type).upper() == "CORRECTIONS"

    # County filter: only when we have a county AND the agency isn't CORRECTIONS
    # (corrections officers move between facilities statewide).
    if source_county and not is_corrections:
        post = filter_by_county(post, source_county)
        if len(post) == 0:
            return post

    # Agency-type mask: only when the data has real type information.
    if has_real_agency_type(post):
        post = filter_by_agency_type(post, agency_type)
        if len(post) == 0:
            return post

    date_mask = in_date_range(post, incident_year)
    name_filtered = filter_by_name(post[date_mask], first_name, last_name)
    return name_filtered
