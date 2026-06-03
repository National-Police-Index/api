"""Tests for the precision-critical scoring-stage logic.

These are the gates that decide auto-match vs review, factored out of the legacy
find_canonical_stint so they can be tested without the ML model or the network:
  - probability threshold
  - exact first+last name gate
  - ambiguity guard (>=2 distinct persons with an exact-name match)
  - best-match selection (dedup by person, then highest prob per mention)
"""
import datetime

import pandas as pd

from shared.models import OfficerMention
from resolve.pipeline import (
    apply_threshold,
    has_exact_name_match,
    split_by_exact_name,
    find_ambiguous_uids,
    select_best_matches,
    _attach_mention,
)


def test_attach_mention_includes_all_name_columns_featurize_needs():
    # featurize() reads mention_{first,middle,last,suffix}_name; all must be present
    # or scoring blows up with a KeyError (regression: mention_suffix was missing).
    mention = OfficerMention(
        mention_uid="m1", mention_first_name="Scott", mention_last_name="Lunger",
        mention_incident_date=datetime.date(2015, 1, 1), state="CA",
    )
    out = _attach_mention(pd.DataFrame([{"post_person_nbr": "P1"}]), mention)
    for col in ("mention_first_name", "mention_middle_name", "mention_last_name", "mention_suffix"):
        assert col in out.columns


def _cand(uid, person, prob, mf="Scott", ml="Lunger", pf="Scott", pl="Lunger"):
    return dict(
        mention_uid=uid, post_person_nbr=person, match_probability=prob,
        mention_first_name=mf, mention_last_name=ml,
        post_first_name=pf, post_last_name=pl,
    )


class TestThreshold:
    def test_keeps_above_threshold(self):
        df = pd.DataFrame([_cand("m1", "P1", 0.9), _cand("m1", "P2", 0.3)])
        out = apply_threshold(df, threshold=0.5)
        assert set(out["post_person_nbr"]) == {"P1"}


class TestExactName:
    def test_exact_match_true(self):
        row = pd.Series(_cand("m1", "P1", 0.9))
        assert has_exact_name_match(row) is True

    def test_case_insensitive(self):
        row = pd.Series(_cand("m1", "P1", 0.9, mf="scott", pf="SCOTT"))
        assert has_exact_name_match(row) is True

    def test_mismatch_false(self):
        row = pd.Series(_cand("m1", "P1", 0.9, pf="Scotty"))
        assert has_exact_name_match(row) is False

    def test_split_separates_exact_from_failed(self):
        df = pd.DataFrame([
            _cand("m1", "P1", 0.9),                 # exact
            _cand("m2", "P2", 0.9, pf="Michael"),   # not exact
        ])
        exact, failed = split_by_exact_name(df)
        assert set(exact["mention_uid"]) == {"m1"}
        assert set(failed["mention_uid"]) == {"m2"}


class TestAmbiguityGuard:
    def test_flags_two_distinct_persons_same_name(self):
        df = pd.DataFrame([_cand("m1", "P1", 0.9), _cand("m1", "P2", 0.8)])
        assert find_ambiguous_uids(df) == {"m1"}

    def test_single_person_not_ambiguous(self):
        df = pd.DataFrame([_cand("m1", "P1", 0.9), _cand("m1", "P1", 0.7)])
        assert find_ambiguous_uids(df) == set()


class TestBestMatchSelection:
    def test_picks_highest_probability_per_mention(self):
        df = pd.DataFrame([
            _cand("m1", "P1", 0.7),
            _cand("m1", "P1", 0.95),  # same person, higher prob
        ])
        best = select_best_matches(df)
        assert len(best) == 1
        assert best.iloc[0]["match_probability"] == 0.95

    def test_one_row_per_mention(self):
        df = pd.DataFrame([
            _cand("m1", "P1", 0.9),
            _cand("m2", "P3", 0.8, mf="Jane", ml="Doe", pf="Jane", pl="Doe"),
        ])
        best = select_best_matches(df)
        assert set(best["mention_uid"]) == {"m1", "m2"}
        assert len(best) == 2
