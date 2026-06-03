"""Group annotated candidates into concentric gate sections for display.

Most central first: the candidate that cleared every gate (the validated auto-match),
then exact-name + above-threshold, then above-threshold but non-exact, then below
threshold. Each candidate lands in exactly one section.
"""
from __future__ import annotations

from typing import List


def _passed_all(c) -> bool:
    return bool(c.get("is_best") and c.get("agency_valid") and c.get("exact_name")
               and c.get("above_threshold"))


def gate_sections(result) -> List[dict]:
    """Return ordered, non-empty sections: [{"title": str, "candidates": [...]}].

    Accepts a MentionResult (uses `.candidates`) or a raw list of annotated candidates.
    """
    candidates = getattr(result, "candidates", result) or []

    buckets = [
        ("✓ Passed ALL gates (auto-match)", _passed_all),
        ("Exact name + above threshold", lambda c: c.get("exact_name") and c.get("above_threshold")),
        ("Above threshold — name not exact", lambda c: c.get("above_threshold") and not c.get("exact_name")),
        ("Below probability threshold", lambda c: not c.get("above_threshold")),
    ]

    sections = []
    claimed = set()
    for title, pred in buckets:
        group = []
        for i, c in enumerate(candidates):
            if i in claimed:
                continue
            if pred(c):
                group.append(c)
                claimed.add(i)
        if group:
            sections.append({"title": title, "candidates": group})
    return sections
