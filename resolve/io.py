"""Input/output for the entity-resolution pipeline.

Reading mention CSVs, deriving required columns, building OfficerMention objects,
and writing the result files. Kept free of model/LLM imports so input prep is cheap
to test. (Output writers are added test-first as the pipeline is consolidated.)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from datetime import datetime
from typing import List, Optional

import pandas as pd

_DATA_INPUT = os.path.join(os.path.dirname(__file__), "data", "input")
_COMMON_LAST_NAMES = os.path.join(_DATA_INPUT, "common_last_names.csv")

# Review reasons that mean "skipped before entity resolution" (early-filtered).
_EARLY_FILTER_MARKERS = ("Common last name", "Multiple persons")


def generate_officer_uid(row: pd.Series) -> str:
    """Stable SHA256 uid for an officer mention, from identifying fields.

    Behavior-preserving port of the legacy match.py helper.
    """
    first_name = str(row.get("first_name", "")).strip()
    last_name = str(row.get("last_name", "")).strip()
    provisional_case_name = str(row.get("provisional_case_name", "")).strip()
    incident_year = str(row.get("incident_year", "")).strip()
    incident_month = str(row.get("incident_month", "")).strip()
    incident_date = str(row.get("incident_date", "")).strip()
    source_agency = str(row.get("source_agency", "")).strip()

    combined = (
        f"{first_name}|{last_name}|{provisional_case_name}|{incident_year}"
        f"|{incident_month}|{incident_date}|{source_agency}"
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _extract_most_recent_year(date_str) -> int | None:
    """Year of the most recent date in a (possibly comma-separated) date string."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None

    dates = [d.strip() for d in str(date_str).strip().split(",") if d.strip()]
    if not dates:
        return None

    parsed = []
    for d in dates:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y", "%Y"):
            try:
                parsed.append(datetime.strptime(d, fmt))
                break
            except ValueError:
                continue

    if not parsed:
        # Fall back to any 4-digit year token.
        for d in dates:
            m = re.search(r"\b(19|20)\d{2}\b", d)
            if m:
                return int(m.group())
        return None

    return max(parsed).year


def ensure_incident_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an `incident_year` column, deriving it from `incident_date` when
    absent. Raises ValueError if neither column exists.

    Behavior-preserving port of the legacy helpers.ensure_incident_year_column.
    """
    if "incident_year" in df.columns:
        return df
    if "incident_date" not in df.columns:
        raise ValueError(
            "Neither 'incident_year' nor 'incident_date' column found in DataFrame"
        )

    df = df.copy()
    df["incident_year"] = df["incident_date"].apply(_extract_most_recent_year)
    return df


def load_common_last_names(path: str = _COMMON_LAST_NAMES) -> set:
    """Set of common last names (uppercased) for Stage-0 early filtering."""
    df = pd.read_csv(path)
    return set(df["last_name"].astype(str).str.strip().str.upper())


def build_mention(row, default_state: Optional[str] = None):
    """Build an OfficerMention from a dict/Series. Names are uppercased (matching
    the legacy pipeline); a uid is generated when absent; state falls back to
    `default_state`. Requires `incident_year` (use ensure_incident_year_column first)."""
    from shared.models import OfficerMention

    def g(key, default=""):
        val = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
        return default if val is None else val

    def up(key):
        v = g(key, "")
        return str(v).upper() if v not in ("", None) and not pd.isna(v) else ""

    uid = g("officer_uid", "") or generate_officer_uid(pd.Series(dict(row)))
    year = int(float(g("incident_year")))
    state = g("state", "") or default_state or None

    return OfficerMention(
        mention_uid=str(uid),
        mention_first_name=up("first_name"),
        mention_middle_name=up("middle_name") or None,
        mention_last_name=up("last_name"),
        mention_suffix=up("suffix") or None,
        mention_agency=g("source_agency", "") or None,
        mention_agency_type=str(g("agency_type", "POLICE") or "POLICE"),
        mention_incident_date=_dt.date(year, 1, 1),
        state=state,
        mentioned_agencies=str(g("mentioned_agencies", "") or ""),
    )


def read_mentions(path: str, default_state: Optional[str] = None,
                  sample_n: Optional[int] = None, sample_seed: Optional[int] = None) -> List:
    """Read a mentions CSV into a list of OfficerMention objects."""
    df = pd.read_csv(path)
    if sample_n:
        df = df.sample(n=sample_n, random_state=sample_seed)
    df = ensure_incident_year_column(df)
    df = df.fillna("")
    df = df[df["incident_year"] != ""]
    return [build_mention(r, default_state=default_state) for _, r in df.iterrows()]


def _input_officer_dict(mention) -> dict:
    return {
        "officer_uid": mention.mention_uid,
        "first_name": mention.mention_first_name,
        "middle_name": mention.mention_middle_name or "",
        "last_name": mention.mention_last_name,
        "suffix": mention.mention_suffix or "",
        "source_agency": mention.mention_agency or "",
        "incident_date": str(mention.mention_incident_date),
        "state": mention.state or "",
        "mentioned_agencies": mention.mentioned_agencies or "",
    }


def _is_early_filtered(reason: str) -> bool:
    return any(m in (reason or "") for m in _EARLY_FILTER_MARKERS)


def write_outputs(results: List, output_dir: str) -> dict:
    """Write the pipeline result set: three JSONL buckets + a flat CSV.

    Buckets mirror the legacy pipeline: auto_matched, early_filtered (common name /
    multiple persons), failed_entity_resolution (everything else routed to review).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "auto_matched": os.path.join(output_dir, "auto_matched.jsonl"),
        "early_filtered": os.path.join(output_dir, "early_filtered.jsonl"),
        "failed_entity_resolution": os.path.join(output_dir, "failed_entity_resolution.jsonl"),
        "csv": os.path.join(output_dir, "results.csv"),
    }

    auto, early, failed, rows = [], [], [], []
    for r in results:
        io_dict = _input_officer_dict(r.mention)
        if r.status == "auto_matched":
            auto.append({"input_officer": io_dict, "post_match": r.match or {}})
            rows.append({**io_dict, "status": "auto_matched",
                         "post_person_nbr": (r.match or {}).get("post_person_nbr", ""),
                         "review_reason": ""})
        elif _is_early_filtered(r.reason):
            early.append({"input_officer": io_dict, "review_reason": r.reason})
            rows.append({**io_dict, "status": "early_filtered",
                         "post_person_nbr": "", "review_reason": r.reason})
        else:
            failed.append({"input_officer": io_dict, "review_reason": r.reason,
                           "candidates": r.candidates})
            rows.append({**io_dict, "status": "review",
                         "post_person_nbr": "", "review_reason": r.reason})

    for key, records in (("auto_matched", auto), ("early_filtered", early),
                         ("failed_entity_resolution", failed)):
        with open(paths[key], "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")

    pd.DataFrame(rows).to_csv(paths["csv"], index=False)
    return paths
