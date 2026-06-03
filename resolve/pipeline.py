"""Entity-resolution pipeline: scoring-stage logic and the PostMatcher orchestrator.

The precision-critical stage functions (threshold, exact-name gate, ambiguity guard,
best-match selection) are pure and import-light so they test without the ML model or
network. The ML scorer and the LLM validator are injected into PostMatcher (defaulting
to the real XGBoost model and the OpenAI agency validator), which keeps the heavy deps
out of unit tests and makes the pipeline reusable as a library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from resolve.candidates import select_candidates

DEFAULT_THRESHOLD = 0.5


# ---- stage 0: early filter (pure decision) --------------------------------------

def stage0_gates(
    mention,
    common_last_names,
    same_name_count: int,
    require_state: bool = True,
) -> tuple[list, bool, str]:
    """Run the Stage-0 (pre-candidate) gates and record each as a checklist entry.

    Returns (gates, should_skip, reason). `gates` is an ordered list of
    {name, status, detail} where status is "pass" (cleared) or "flag" (route to review)
    or "fail" (missing state). Stops appending at the first gate that routes to review.
    """
    def _get(key):
        if isinstance(mention, dict):
            return mention.get(key)
        return getattr(mention, key, None)

    state = _get("mention_state") or _get("state")
    last = (_get("mention_last_name") or "").strip().upper()
    gates: list = []

    if require_state:
        ok = bool(state)
        gates.append({"name": "State present", "status": "pass" if ok else "fail",
                      "detail": state or "(none)"})
        if not ok:
            return gates, True, "No state provided - cannot safely resolve against all-states table"

    common = last in common_last_names
    gates.append({"name": "Common last name", "status": "flag" if common else "pass",
                  "detail": last if common else ""})
    if common:
        return gates, True, f"Common last name ({last}) - requires manual verification"

    multi = same_name_count >= 2
    gates.append({"name": "Unique name in state", "status": "flag" if multi else "pass",
                  "detail": f"{same_name_count} person(s)"})
    if multi:
        return gates, True, f"Multiple persons ({same_name_count}) with same name - needs verification"

    return gates, False, ""


def early_filter_decision(mention, common_last_names, same_name_count, require_state=True):
    """Thin wrapper over stage0_gates returning just (should_skip, reason)."""
    _gates, skip, reason = stage0_gates(mention, common_last_names, same_name_count, require_state)
    return skip, reason


# ---- scoring-stage logic (pure) -------------------------------------------------

def apply_threshold(candidates: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Keep candidates whose match_probability exceeds the threshold."""
    if len(candidates) == 0:
        return candidates
    return candidates[candidates["match_probability"] > threshold]


def has_exact_name_match(row) -> bool:
    """True if first AND last name match exactly (case-insensitive, trimmed)."""
    mf = str(row.get("mention_first_name", "")).strip().upper()
    ml = str(row.get("mention_last_name", "")).strip().upper()
    pf = str(row.get("post_first_name", "")).strip().upper()
    pl = str(row.get("post_last_name", "")).strip().upper()
    return (mf == pf) and (ml == pl)


def split_by_exact_name(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (exact-name matches, failed) — failed get a validation_reason."""
    if len(candidates) == 0:
        return candidates, candidates
    mask = candidates.apply(has_exact_name_match, axis=1)
    exact = candidates[mask].copy()
    failed = candidates[~mask].copy()
    if len(failed) > 0:
        failed["validation_reason"] = "High similarity score but no exact first+last name match"
    return exact, failed


def find_ambiguous_uids(candidates: pd.DataFrame) -> set:
    """Mention uids with >=2 distinct persons among their candidates (ambiguous)."""
    if len(candidates) == 0:
        return set()
    counts = candidates.groupby("mention_uid")["post_person_nbr"].nunique()
    return set(counts[counts >= 2].index)


def select_best_matches(candidates: pd.DataFrame) -> pd.DataFrame:
    """One best row per mention: dedup by person (keep highest prob), then by mention."""
    if len(candidates) == 0:
        return candidates
    ordered = candidates.sort_values(
        by=["mention_uid", "match_probability"], ascending=[True, False]
    )
    best_per_person = ordered.drop_duplicates(
        subset=["mention_uid", "post_person_nbr"], keep="first"
    )
    return best_per_person.drop_duplicates(subset="mention_uid", keep="first")


# ---- orchestrator ---------------------------------------------------------------

@dataclass
class MentionResult:
    """Per-mention verdict from the pipeline.

    `candidates` are ALL generated candidates, each annotated with the gates it cleared
    (`above_threshold`, `exact_name`, `is_best`, `agency_valid`) so a UI can render them
    in concentric sections. `ambiguous` flags ≥2 distinct exact-name persons.
    """
    mention: object
    status: str                      # "auto_matched" | "review"
    reason: str = ""
    match: Optional[dict] = None     # matched POST record (auto_matched only)
    candidates: List[dict] = field(default_factory=list)
    ambiguous: bool = False
    gates: List[dict] = field(default_factory=list)  # ordered stage checklist {name,status,detail}


_POST_FIELDS = (
    "post_person_nbr", "post_first_name", "post_middle_name", "post_last_name",
    "post_suffix", "post_agency_name", "post_agency_type", "post_start_date",
    "post_end_date", "post_separation_reason", "state", "county",
)

_GATE_FIELDS = ("match_probability", "above_threshold", "exact_name", "is_best", "agency_valid")


def _attach_mention(df: pd.DataFrame, mention) -> pd.DataFrame:
    """Add the mention's scalar fields as columns (one mention per call)."""
    df = df.copy()
    df["mention_uid"] = mention.mention_uid
    df["mention_first_name"] = mention.mention_first_name
    df["mention_last_name"] = mention.mention_last_name
    df["mention_middle_name"] = getattr(mention, "mention_middle_name", "") or ""
    df["mention_suffix"] = getattr(mention, "mention_suffix", "") or ""
    df["mention_agency"] = getattr(mention, "mention_agency", "") or ""
    df["mention_agency_type"] = str(getattr(mention, "mention_agency_type", "POLICE"))
    df["mentioned_agencies"] = getattr(mention, "mentioned_agencies", "") or ""
    return df.reset_index(drop=True)


def _candidate_dicts(df: pd.DataFrame) -> List[dict]:
    cols = [c for c in (*_POST_FIELDS, *_GATE_FIELDS) if c in df.columns]
    return df[cols].to_dict("records")


def _match_dict(row) -> dict:
    out = {f: row.get(f) for f in _POST_FIELDS if f in row.index}
    if "match_probability" in row.index:
        out["match_probability"] = float(row["match_probability"])
    return out


class PostMatcher:
    """Resolve officer mentions to POST employment records.

    Dependencies are injectable so the pipeline is testable and reusable:
      - client:    fetches candidates / same-name records / county (default: NPIClient)
      - scorer:    candidates DataFrame -> match_probability Series (default: XGBoost)
      - validator: (mention_agency, mentioned_agencies, post_agency) -> (bool, reason)
                   (default: deterministic non-LE guard + LLM)

    `require_state` gates the all-states safety rule (True) vs CA `postie` (may be False).
    """

    def __init__(
        self,
        client=None,
        scorer: Optional[Callable] = None,
        validator: Optional[Callable] = None,
        *,
        api_url: Optional[str] = None,
        require_state: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        common_last_names=None,
    ):
        self._client = client
        self._api_url = api_url
        self._scorer = scorer
        self._validator = validator
        self.require_state = require_state
        self.threshold = threshold
        self._common = common_last_names

    # lazy defaults keep heavy deps (model, openai, requests) out of unit tests
    @property
    def client(self):
        if self._client is None:
            from resolve.client import NPIClient
            self._client = NPIClient(base_url=self._api_url)
        return self._client

    @property
    def scorer(self):
        if self._scorer is None:
            from resolve.scoring import xgboost_scorer
            self._scorer = xgboost_scorer
        return self._scorer

    @property
    def validator(self):
        if self._validator is None:
            from resolve.validation import validate_agency_match
            self._validator = validate_agency_match
        return self._validator

    @property
    def common_last_names(self):
        if self._common is None:
            from resolve.io import load_common_last_names
            self._common = load_common_last_names()
        return self._common

    def resolve_one(self, mention) -> MentionResult:
        return self.resolve_batch([mention])[0]

    def resolve_batch(self, mentions) -> List[MentionResult]:
        return [self._resolve_single(m) for m in mentions]

    # ---- internals --------------------------------------------------------------
    def _same_name_count(self, mention, state) -> int:
        try:
            recs = self.client.get_officers_by_name(
                mention.mention_first_name, mention.mention_last_name, state=state
            )
        except Exception:
            return 0
        persons = set()
        for r in recs:
            d = r.dict() if hasattr(r, "dict") else r
            persons.add(d.get("post_person_nbr") if isinstance(d, dict) else getattr(r, "post_person_nbr", None))
        return len(persons)

    def _resolve_single(self, mention) -> MentionResult:
        state = getattr(mention, "state", None)

        # Stage 0: early filter (records its own gate checklist entries)
        same_name = self._same_name_count(mention, state)
        gates, skip, reason = stage0_gates(
            {
                "mention_first_name": mention.mention_first_name,
                "mention_last_name": mention.mention_last_name,
                "state": state,
            },
            self.common_last_names,
            same_name,
            require_state=self.require_state,
        )
        if skip:
            return MentionResult(mention, "review", reason, gates=gates)

        # Stage 1: candidate generation
        incident_year = mention.mention_incident_date.year
        _at = getattr(mention, "mention_agency_type", "POLICE")
        agency_type = getattr(_at, "value", _at)  # AgencyType enum -> "POLICE", str passthrough
        api_cands = self.client.get_candidates_for_mention(
            first_name=mention.mention_first_name,
            last_name=mention.mention_last_name,
            incident_year=incident_year,
            state=state,
            agency_type=agency_type,
        )
        post = pd.DataFrame([c.dict() if hasattr(c, "dict") else c for c in api_cands]) \
            if api_cands else pd.DataFrame()

        # Optional county lookup (skipped for CORRECTIONS or when unavailable)
        source_county = None
        if api_cands and getattr(mention, "mention_agency", None) and agency_type.upper() != "CORRECTIONS":
            try:
                source_county = self.client.get_county_for_agency(mention.mention_agency)
            except Exception:
                source_county = None

        filtered = select_candidates(
            post, mention.mention_first_name, mention.mention_last_name, incident_year,
            agency_type=agency_type, source_county=source_county,
        ) if len(post) else post

        gates.append({"name": "Candidates found", "status": "pass" if len(filtered) else "fail",
                      "detail": f"{len(filtered)} candidate(s)"})
        if len(filtered) == 0:
            return MentionResult(mention, "review", "No candidates found", gates=gates)

        # Stage 2: scoring + per-candidate gate annotation
        filtered = _attach_mention(filtered, mention)
        filtered["match_probability"] = list(self.scorer(filtered))
        filtered["above_threshold"] = filtered["match_probability"] > self.threshold
        filtered["exact_name"] = filtered.apply(has_exact_name_match, axis=1)
        filtered["is_best"] = False
        filtered["agency_valid"] = None

        # Eligible = cleared the probability threshold AND the exact first+last name gate.
        eligible = filtered[filtered["above_threshold"] & filtered["exact_name"]]
        n_exact_persons = int(eligible["post_person_nbr"].nunique())
        ambiguous = n_exact_persons >= 2

        gates.append({"name": "Exact-name match", "status": "pass" if len(eligible) else "fail",
                      "detail": f"{len(eligible)} eligible candidate(s)"})
        gates.append({"name": "Not ambiguous",
                      "status": "flag" if ambiguous else ("pass" if len(eligible) else "skip"),
                      "detail": f"{n_exact_persons} exact-name person(s)"})

        status, reason, match = "review", "", None
        agency_status, agency_detail = "skip", ""
        if len(eligible) == 0:
            reason = "High similarity score but no exact first+last name match"
        elif ambiguous:
            reason = ("Multiple distinct persons with exact name match in state - "
                      "ambiguous (no auto-match)")
        else:
            best = select_best_matches(eligible)
            best_person = best.iloc[0]["post_person_nbr"]
            # mark every stint of the selected person as the best
            filtered.loc[filtered["post_person_nbr"] == best_person, "is_best"] = True

            # Stage 4: agency validation (on the selected best)
            valid, vreason = self.validator(
                best.iloc[0].get("mention_agency", ""),
                best.iloc[0].get("mentioned_agencies", ""),
                best.iloc[0].get("post_agency_name", ""),
            )
            filtered.loc[filtered["post_person_nbr"] == best_person, "agency_valid"] = valid
            if valid:
                status, match = "auto_matched", _match_dict(best.iloc[0])
                agency_status = "pass"
            else:
                reason = vreason or "Agency cannot be validated"
                agency_status, agency_detail = "fail", reason

        gates.append({"name": "Agency validation", "status": agency_status, "detail": agency_detail})

        cand_list = _candidate_dicts(
            filtered.sort_values(["is_best", "match_probability"], ascending=[False, False])
        )
        return MentionResult(mention, status, reason, match=match,
                             candidates=cand_list, ambiguous=bool(ambiguous), gates=gates)
