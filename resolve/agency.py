"""Deterministic agency classification + the non-LE guard.

This module is import-clean: it pulls in NO LLM client, so the precision-critical
guard can be unit-tested without an OpenAI key or network. The LLM-backed
validation that uses these helpers lives in `resolve.validation`.

Ported verbatim (behavior-preserving) from the legacy `resolve/src/helpers.py`.
"""
from __future__ import annotations

import ast
from typing import List, Tuple

# Keywords that mark a source agency as non-law-enforcement (prosecutorial,
# coronial, defense, etc.). These appear as `source_agency` in incident reports
# but must never auto-match a POST police/sheriff record without a corroborating
# LE agency in `mentioned_agencies`.
NON_LE_KEYWORDS = (
    "district attorney",
    "attorney general",
    "public defender",
    "coroner",
    "medical examiner",
    " da ",
    " me ",
    "office of the da",
)

# Keywords that mark a POST/agency string as a law-enforcement employer.
LE_KEYWORDS = (
    "police",
    "sheriff",
    "marshal",
    "patrol",
    "highway patrol",
    "probation",
    "corrections",
    "department of public safety",
    "public safety",
)

_NON_LE_GUARD_REASON = (
    "Source agency is non-LE (DA/Coroner/ME/etc.) with no mentioned "
    "LE agencies; cannot auto-validate against an LE POST agency"
)


def is_non_le_agency(name: str) -> bool:
    """True if the agency name is non-LE. A name containing any LE keyword is LE
    (so mixed strings like "Sheriff's Office / DA" count as LE, not non-LE)."""
    if not name:
        return False
    n = f" {name.lower().strip()} "
    if any(kw in n for kw in LE_KEYWORDS):
        return False
    return any(kw in n for kw in NON_LE_KEYWORDS)


def is_le_agency(name: str) -> bool:
    """True if the agency name contains a law-enforcement keyword."""
    if not name:
        return False
    return any(kw in name.lower() for kw in LE_KEYWORDS)


def all_non_le(agencies: List[str]) -> bool:
    """True iff the list is non-empty and every agency is non-LE."""
    return bool(agencies) and all(is_non_le_agency(a) for a in agencies)


def parse_agencies_to_check(mention_agency: str, mentioned_agencies) -> List[str]:
    """Build the list of agencies to compare a POST agency against: the source
    `mention_agency` plus every entry in `mentioned_agencies` (a stringified list
    or a plain string)."""
    agencies: List[str] = []
    if mention_agency:
        agencies.append(mention_agency)

    if mentioned_agencies and str(mentioned_agencies).strip():
        try:
            parsed = (
                ast.literal_eval(mentioned_agencies)
                if isinstance(mentioned_agencies, str)
                else mentioned_agencies
            )
            if isinstance(parsed, list):
                agencies.extend(str(a) for a in parsed if a)
            else:
                agencies.append(str(mentioned_agencies))
        except (ValueError, SyntaxError):
            agencies.append(str(mentioned_agencies))

    return agencies


def non_le_guard(agencies_to_check: List[str], post_agency: str) -> Tuple[bool, str]:
    """Deterministic pre-LLM guard.

    Returns (blocked, reason). Blocks when every agency to compare against is
    non-LE and the POST agency is LE — bypassed if any LE agency is present.
    """
    if all_non_le(agencies_to_check) and is_le_agency(post_agency):
        return True, _NON_LE_GUARD_REASON
    return False, ""
