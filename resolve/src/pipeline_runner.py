"""
pipeline_runner.py — run the entity-resolution pipeline on ONE mention, in a SEPARATE
process, and write a compact JSON result.

The TUI (explore_tui.py) invokes this as a subprocess so that ALL of the pipeline's noisy
output — Python prints AND the C-level writes from torch / transformers / tqdm / warnings,
which go straight to file descriptors 1/2 — is captured by the parent into a log file and
can NEVER bleed onto the Textual terminal. (Redirecting sys.stdout in-process doesn't catch
the fd-level writes; a child process does, cleanly.)

Usage:
    python pipeline_runner.py <mention_json_path> <result_json_path>

The mention JSON has keys: uid, first, last, middle, state, year, agency, mentioned.
The result JSON has: {matched: [...], reason: str|null, candidates: [...]}.
All human-readable noise goes to this process's stdout/stderr (captured by the parent).
"""
import os
import sys
import json
import datetime

# Same path setup as explore_tui.py: this dir for siblings (api, match_all_states, features),
# repo root for `models.src`. cwd is set to this dir by the parent (relative model/CSV paths).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def _d(v):
    if v is None:
        return ""
    s = str(v)
    return s[:10] if s and s != "NaT" else ""


def main() -> int:
    mention_path, result_path = sys.argv[1], sys.argv[2]
    with open(mention_path) as f:
        m = json.load(f)

    # Heavy imports happen here, in the child — their noise is the parent's problem to capture.
    from models.src import OfficerMention
    from match_all_states import PostMatcher

    mention = OfficerMention(
        mention_uid=m["uid"],
        mention_agency_type="POLICE",
        mention_incident_date=datetime.date(int(m["year"]), 1, 1),
        mention_first_name=m["first"].upper(),
        mention_middle_name=(m.get("middle") or "").upper() or None,
        mention_last_name=m["last"].upper(),
        mention_agency=m.get("agency") or None,
        state=m["state"],
        mentioned_agencies=m.get("mentioned") or "",
    )

    matcher = PostMatcher()
    matched, all_cands, invalid, _hist = matcher.find_canonical_stint([mention])
    uid = m["uid"]

    result = {"matched": [], "reason": None, "candidates": []}

    if matched is not None and len(matched) > 0:
        row = matched.iloc[0]
        result["matched"].append({
            "post_first_name": str(row.get("post_first_name", "")),
            "post_last_name": str(row.get("post_last_name", "")),
            "post_agency_name": str(row.get("post_agency_name", "")),
            "post_person_nbr": str(row.get("post_person_nbr", "")),
            "match_probability": float(row.get("match_probability", 0.0)),
        })
    else:
        reason = "No candidates found"
        if invalid is not None and len(invalid) > 0:
            rmatch = invalid[invalid["mention_uid"] == uid]
            if len(rmatch) > 0:
                reason = str(rmatch.iloc[0].get("validation_reason", reason))
        result["reason"] = reason

    if all_cands is not None and len(all_cands) > 0:
        cc = all_cands[all_cands["mention_uid"] == uid] if "mention_uid" in all_cands else all_cands
        if "match_probability" in cc:
            cc = cc.sort_values("match_probability", ascending=False)
        for _, c in cc.iterrows():
            prob = c.get("match_probability")
            result["candidates"].append({
                "post_person_nbr": str(c.get("post_person_nbr", "")),
                "post_first_name": str(c.get("post_first_name", "")),
                "post_last_name": str(c.get("post_last_name", "")),
                "post_agency_name": str(c.get("post_agency_name", "")),
                "post_start_date": _d(c.get("post_start_date")),
                "post_end_date": _d(c.get("post_end_date")),
                "match_probability": float(prob) if prob is not None else None,
            })

    with open(result_path, "w") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
