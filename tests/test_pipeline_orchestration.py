"""End-to-end orchestration of PostMatcher with injected fakes (no model, no network).

Exercises resolve_one / resolve_batch across the decision paths:
auto-match, common-name early filter, ambiguity guard, agency-validation reject,
and no-candidates -> review.
"""
import datetime

import pandas as pd
import pytest

from shared.models import OfficerMention
from resolve.pipeline import PostMatcher, MentionResult


def make_mention(uid="m1", first="Scott", last="Lunger", state="CA",
                 agency="Hayward Police Department", year=2015, mentioned="[]"):
    return OfficerMention(
        mention_uid=uid,
        mention_first_name=first,
        mention_last_name=last,
        mention_agency=agency,
        mention_agency_type="POLICE",
        mention_incident_date=datetime.date(year, 1, 1),
        state=state,
        mentioned_agencies=mentioned,
    )


class FakeClient:
    """Stand-in for NPIClient. `candidates` is a list of dicts; `same_name` controls
    the distinct-person count returned by get_officers_by_name."""

    def __init__(self, candidates=None, same_name_persons=("P1",), county=None):
        self._candidates = candidates or []
        self._same_name = same_name_persons
        self._county = county

    def get_candidates_for_mention(self, first_name, last_name, incident_year,
                                   state=None, agency_type="POLICE"):
        self.last_agency_type = agency_type
        return [_Rec(**c) for c in self._candidates]

    def get_officers_by_name(self, first_name, last_name, state=None):
        return [_Rec(post_person_nbr=p, post_first_name=first_name,
                     post_last_name=last_name) for p in self._same_name]

    def get_county_for_agency(self, agency_name):
        return self._county


class _Rec:
    """Minimal record object exposing .dict() like a Pydantic model."""
    _DEFAULTS = dict(
        post_person_nbr="P1", post_first_name="Scott", post_middle_name="",
        post_last_name="Lunger", post_suffix="", post_agency_name="Hayward Police Department",
        post_agency_type="POLICE", post_start_date="2010-01-01", post_end_date="2020-01-01",
        post_separation_reason="", state="CA", county=None,
    )

    def __init__(self, **kw):
        self._d = {**self._DEFAULTS, **kw}

    def dict(self):
        return dict(self._d)


def exact_name_scorer(candidates: pd.DataFrame) -> pd.Series:
    """Fake scorer: high prob when names match exactly, else low."""
    def score(row):
        return 0.95 if str(row["post_first_name"]).upper() == str(row["mention_first_name"]).upper() \
            and str(row["post_last_name"]).upper() == str(row["mention_last_name"]).upper() else 0.2
    return candidates.apply(score, axis=1)


def always_valid(mention_agency, mentioned_agencies, post_agency):
    return True, ""


def always_invalid(mention_agency, mentioned_agencies, post_agency):
    return False, "agency mismatch"


def _matcher(client, validator=always_valid):
    return PostMatcher(client=client, scorer=exact_name_scorer, validator=validator,
                       common_last_names={"SMITH", "JOHNSON"})


class TestResolveOne:
    def test_clean_mention_auto_matches(self):
        client = FakeClient(candidates=[dict(post_person_nbr="P1")], same_name_persons=("P1",))
        result = _matcher(client).resolve_one(make_mention())
        assert isinstance(result, MentionResult)
        assert result.status == "auto_matched"
        assert result.match["post_person_nbr"] == "P1"

    def test_common_last_name_routed_to_review(self):
        client = FakeClient(candidates=[dict(post_person_nbr="P1")])
        result = _matcher(client).resolve_one(make_mention(last="Smith"))
        assert result.status == "review"
        assert "common" in result.reason.lower()

    def test_multiple_persons_same_name_routed(self):
        client = FakeClient(candidates=[dict(post_person_nbr="P1")],
                            same_name_persons=("P1", "P2"))
        result = _matcher(client).resolve_one(make_mention())
        assert result.status == "review"
        assert "multiple" in result.reason.lower()

    def test_ambiguity_guard_two_exact_persons(self):
        # Two distinct persons, both exact-name matches -> ambiguous -> review.
        # same_name scoped lookup returns one (passes stage 0) but candidate set has two.
        client = FakeClient(
            candidates=[dict(post_person_nbr="P1"), dict(post_person_nbr="P2")],
            same_name_persons=("P1",),
        )
        result = _matcher(client).resolve_one(make_mention())
        assert result.status == "review"

    def test_agency_validation_reject_routes_to_review(self):
        client = FakeClient(candidates=[dict(post_person_nbr="P1")])
        result = _matcher(client, validator=always_invalid).resolve_one(make_mention())
        assert result.status == "review"
        assert "agency" in result.reason.lower()

    def test_no_candidates_routes_to_review(self):
        client = FakeClient(candidates=[])
        result = _matcher(client).resolve_one(make_mention())
        assert result.status == "review"
        assert "candidate" in result.reason.lower()


class TestAgencyTypeIsPassedAsValue:
    def test_enum_agency_type_sent_as_plain_string(self):
        # Regression: mention.mention_agency_type is an AgencyType enum; the client must
        # receive "POLICE", not "AgencyType.POLICE" (which the API rejects with 422).
        client = FakeClient(candidates=[dict(post_person_nbr="P1")])
        _matcher(client).resolve_one(make_mention())
        assert client.last_agency_type == "POLICE"


class TestResolveBatch:
    def test_returns_one_result_per_mention(self):
        client = FakeClient(candidates=[dict(post_person_nbr="P1")])
        mentions = [make_mention(uid="m1"), make_mention(uid="m2", last="Smith")]
        results = _matcher(client).resolve_batch(mentions)
        assert len(results) == 2
        by_uid = {r.mention.mention_uid: r for r in results}
        assert by_uid["m1"].status == "auto_matched"
        assert by_uid["m2"].status == "review"
