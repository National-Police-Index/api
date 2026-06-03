"""Per-candidate gate annotations + the sectioned report grouping.

Each generated candidate is annotated with which gates it cleared (above_threshold,
exact_name, is_best, agency_valid) so the UI can show concentric sections: the
all-gates-passing match at the center, then candidates that pass fewer gates.
"""
import datetime

import pandas as pd

from shared.models import OfficerMention
from resolve.pipeline import PostMatcher, MentionResult
from resolve.explain import gate_sections


def mention(uid="m1"):
    return OfficerMention(
        mention_uid=uid, mention_first_name="Scott", mention_last_name="Lunger",
        mention_agency="Hayward Police Department", mention_agency_type="POLICE",
        mention_incident_date=datetime.date(2015, 1, 1), state="CA", mentioned_agencies="[]",
    )


class _Rec:
    _D = dict(post_person_nbr="P1", post_first_name="Scott", post_middle_name="",
              post_last_name="Lunger", post_suffix="", post_agency_name="Hayward Police Department",
              post_agency_type="POLICE", post_start_date="2010-01-01", post_end_date="2020-01-01",
              post_separation_reason="", state="CA", county=None)

    def __init__(self, **kw):
        self._d = {**self._D, **kw}

    def dict(self):
        return dict(self._d)


class FakeClient:
    def __init__(self, candidates, same=("P1",)):
        self._c = candidates
        self._same = same

    def get_candidates_for_mention(self, **kw):
        return [_Rec(**c) for c in self._c]

    def get_officers_by_name(self, first_name, last_name, state=None):
        return [_Rec(post_person_nbr=p) for p in self._same]

    def get_county_for_agency(self, agency_name):
        return None


def prob_scorer(mapping, default=0.2):
    def score(df):
        return df["post_person_nbr"].map(lambda p: mapping.get(p, default))
    return score


def matcher(client, scorer, validator=lambda *a: (True, "")):
    return PostMatcher(client=client, scorer=scorer, validator=validator,
                       common_last_names=set())


class TestCandidateAnnotation:
    def test_flags_present_on_every_candidate(self):
        client = FakeClient([
            dict(post_person_nbr="P1", post_last_name="Lunger"),     # exact, high
            dict(post_person_nbr="P3", post_last_name="Lunardi"),    # non-exact, high
        ])
        res = matcher(client, prob_scorer({"P1": 0.9, "P3": 0.8})).resolve_one(mention())
        assert res.status == "auto_matched"
        by = {c["post_person_nbr"]: c for c in res.candidates}
        # exact, above threshold, selected, agency-validated
        assert by["P1"]["exact_name"] is True
        assert by["P1"]["above_threshold"] is True
        assert by["P1"]["is_best"] is True
        assert by["P1"]["agency_valid"] is True
        # non-exact name surfaced but flagged as not exact / not best
        assert by["P3"]["exact_name"] is False
        assert by["P3"]["is_best"] is False
        assert by["P3"]["agency_valid"] is None

    def test_below_threshold_flagged(self):
        client = FakeClient([
            dict(post_person_nbr="P1", post_last_name="Lunger"),     # exact, high
            dict(post_person_nbr="P5", post_last_name="Lunger"),     # exact, low -> below threshold
        ])
        res = matcher(client, prob_scorer({"P1": 0.9, "P5": 0.3})).resolve_one(mention())
        by = {c["post_person_nbr"]: c for c in res.candidates}
        assert by["P5"]["above_threshold"] is False
        assert by["P5"]["exact_name"] is True


class TestGateSections:
    def test_sections_ordered_most_to_least_central(self):
        client = FakeClient([
            dict(post_person_nbr="P1", post_last_name="Lunger"),     # all gates
            dict(post_person_nbr="P3", post_last_name="Lunardi"),    # exact? no; above thresh
            dict(post_person_nbr="P5", post_last_name="Lunger"),     # exact; below thresh
        ])
        res = matcher(client, prob_scorer({"P1": 0.9, "P3": 0.8, "P5": 0.3})).resolve_one(mention())
        sections = gate_sections(res)
        titles = [s["title"] for s in sections]
        # the auto-match section is first and contains P1
        assert "auto" in titles[0].lower() or "all gate" in titles[0].lower()
        first_ids = [c["post_person_nbr"] for c in sections[0]["candidates"]]
        assert first_ids == ["P1"]
        # every surfaced candidate appears in exactly one section
        all_ids = [c["post_person_nbr"] for s in sections for c in s["candidates"]]
        assert sorted(all_ids) == ["P1", "P3", "P5"]

    def test_ambiguous_has_no_automatch_section(self):
        client = FakeClient([
            dict(post_person_nbr="P1", post_last_name="Lunger"),
            dict(post_person_nbr="P2", post_last_name="Lunger"),     # 2nd exact person -> ambiguous
        ])
        res = matcher(client, prob_scorer({"P1": 0.9, "P2": 0.85})).resolve_one(mention())
        assert res.status == "review"
        assert res.ambiguous is True
        sections = gate_sections(res)
        # no candidate is marked the validated auto-match
        assert all(not (c.get("is_best") and c.get("agency_valid")) for s in sections for c in s["candidates"])


class TestPipelineGatesChecklist:
    def _common_matcher(self, client, scorer, common=("SMITH",), validator=lambda *a: (True, "")):
        return PostMatcher(client=client, scorer=scorer, validator=validator,
                           common_last_names=set(common))

    def test_clean_match_records_all_stage_gates_passed(self):
        client = FakeClient([dict(post_person_nbr="P1", post_last_name="Lunger")])
        res = self._common_matcher(client, prob_scorer({"P1": 0.9})).resolve_one(mention())
        names = {g["name"]: g["status"] for g in res.gates}
        assert names.get("State present") == "pass"
        assert names.get("Common last name") == "pass"
        assert names.get("Unique name in state") == "pass"
        assert names.get("Candidates found") == "pass"
        assert names.get("Exact-name match") == "pass"
        assert names.get("Not ambiguous") == "pass"
        assert names.get("Agency validation") == "pass"

    def test_common_last_name_flagged_and_stops_early(self):
        client = FakeClient([dict(post_person_nbr="P1")])
        m = OfficerMention(
            mention_uid="m1", mention_first_name="John", mention_last_name="Smith",
            mention_agency="X PD", mention_agency_type="POLICE",
            mention_incident_date=datetime.date(2015, 1, 1), state="CA", mentioned_agencies="[]",
        )
        res = self._common_matcher(client, prob_scorer({"P1": 0.9})).resolve_one(m)
        assert res.status == "review"
        g = {x["name"]: x["status"] for x in res.gates}
        assert g["Common last name"] == "flag"
        # stopped before generating candidates
        assert "Candidates found" not in g

    def test_multiple_in_state_flagged(self):
        client = FakeClient([dict(post_person_nbr="P1")], same=("P1", "P2"))
        res = self._common_matcher(client, prob_scorer({"P1": 0.9})).resolve_one(mention())
        assert res.status == "review"
        g = {x["name"]: x["status"] for x in res.gates}
        assert g["Unique name in state"] == "flag"

    def test_ambiguity_flagged_in_gates(self):
        client = FakeClient([
            dict(post_person_nbr="P1", post_last_name="Lunger"),
            dict(post_person_nbr="P2", post_last_name="Lunger"),
        ])
        res = self._common_matcher(client, prob_scorer({"P1": 0.9, "P2": 0.85})).resolve_one(mention())
        g = {x["name"]: x["status"] for x in res.gates}
        assert g["Not ambiguous"] == "flag"
