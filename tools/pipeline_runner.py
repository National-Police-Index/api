"""pipeline_runner.py — resolve ONE mention in a SEPARATE process, write a compact JSON.

The TUI (explore_tui.py) invokes this as a subprocess so all of the pipeline's noisy
output (Python prints AND fd-level writes from torch/transformers/tqdm) is captured by the
parent into a log file and can never bleed onto the Textual terminal.

Usage:
    python pipeline_runner.py <mention_json_path> <result_json_path>

Mention JSON keys: uid, first, last, middle, state, year, agency, mentioned.
Result JSON: {status, reason, ambiguous, match, sections:[{title, candidates:[...]}]}.
Each candidate carries its gate flags (above_threshold, exact_name, is_best, agency_valid).
"""
import json
import os
import sys

# Repo root on path so `resolve` / `shared` import (cwd-independent — the package uses
# __file__-relative paths for its model/data files).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _d(v):
    if v is None:
        return ""
    s = str(v)
    return s[:10] if s and s != "NaT" else ""


def main() -> int:
    mention_path, result_path = sys.argv[1], sys.argv[2]
    with open(mention_path) as f:
        m = json.load(f)

    # Heavy imports happen here, in the child — their noise is the parent's to capture.
    from resolve import PostMatcher, build_mention
    from resolve.explain import gate_sections

    mention = build_mention({
        "officer_uid": m["uid"],
        "first_name": m["first"],
        "last_name": m["last"],
        "middle_name": m.get("middle") or "",
        "source_agency": m.get("agency") or "",
        "mentioned_agencies": m.get("mentioned") or "",
        "incident_year": int(m["year"]),
        "state": m["state"],
    })

    verdict = PostMatcher().resolve_one(mention)

    def _cand(c):
        prob = c.get("match_probability")
        return {
            "post_person_nbr": str(c.get("post_person_nbr", "")),
            "post_first_name": str(c.get("post_first_name", "")),
            "post_last_name": str(c.get("post_last_name", "")),
            "post_agency_name": str(c.get("post_agency_name", "")),
            "post_start_date": _d(c.get("post_start_date")),
            "post_end_date": _d(c.get("post_end_date")),
            "match_probability": float(prob) if prob is not None else None,
            "above_threshold": bool(c.get("above_threshold")),
            "exact_name": bool(c.get("exact_name")),
            "is_best": bool(c.get("is_best")),
            "agency_valid": c.get("agency_valid"),  # True / False / None
        }

    result = {
        "status": verdict.status,
        "reason": verdict.reason or None,
        "ambiguous": bool(verdict.ambiguous),
        "mention": {"first": m["first"], "last": m["last"], "state": m["state"]},
        "gates": verdict.gates,
        "match": _cand(verdict.match) if verdict.match else None,
        "sections": [
            {"title": s["title"], "candidates": [_cand(c) for c in s["candidates"]]}
            for s in gate_sections(verdict)
        ],
    }

    with open(result_path, "w") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
