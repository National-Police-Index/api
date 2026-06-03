"""Tests for mention construction and result output writing."""
import datetime
import json

from shared.models import OfficerMention
from resolve.io import load_common_last_names, build_mention, write_outputs
from resolve.pipeline import MentionResult


class TestLoadCommonLastNames:
    def test_returns_uppercase_set_including_known_common(self):
        names = load_common_last_names()
        assert isinstance(names, set)
        assert "SMITH" in names  # uppercased


class TestBuildMention:
    def test_builds_from_row_with_year(self):
        m = build_mention({
            "first_name": "scott", "last_name": "lunger", "source_agency": "Hayward PD",
            "incident_year": 2015, "state": "CA",
        })
        assert isinstance(m, OfficerMention)
        assert m.mention_first_name == "SCOTT"  # uppercased
        assert m.mention_last_name == "LUNGER"
        assert m.mention_agency == "Hayward PD"
        assert m.state == "CA"
        assert m.mention_incident_date == datetime.date(2015, 1, 1)

    def test_uses_default_state_when_missing(self):
        m = build_mention({"first_name": "A", "last_name": "B", "incident_year": 2015},
                          default_state="CA")
        assert m.state == "CA"

    def test_generates_uid_when_absent(self):
        m = build_mention({"first_name": "A", "last_name": "B", "incident_year": 2015})
        assert m.mention_uid and len(m.mention_uid) == 64

    def test_respects_existing_uid(self):
        m = build_mention({"first_name": "A", "last_name": "B", "incident_year": 2015,
                           "officer_uid": "fixed-uid"})
        assert m.mention_uid == "fixed-uid"


def _mention(uid="m1", last="Lunger"):
    return OfficerMention(
        mention_uid=uid, mention_first_name="Scott", mention_last_name=last,
        mention_agency="Hayward PD", mention_agency_type="POLICE",
        mention_incident_date=datetime.date(2015, 1, 1), state="CA",
        mentioned_agencies="[]",
    )


class TestWriteOutputs:
    def test_writes_three_jsonl_buckets_and_csv(self, tmp_path):
        results = [
            MentionResult(_mention("m1"), "auto_matched", "",
                          match={"post_person_nbr": "P1", "post_agency_name": "Hayward PD"}),
            MentionResult(_mention("m2", last="Smith"), "review",
                          "Common last name (SMITH) - requires manual verification"),
            MentionResult(_mention("m3"), "review", "No candidates found"),
        ]
        paths = write_outputs(results, str(tmp_path))

        auto = [json.loads(l) for l in open(paths["auto_matched"])]
        early = [json.loads(l) for l in open(paths["early_filtered"])]
        failed = [json.loads(l) for l in open(paths["failed_entity_resolution"])]

        assert len(auto) == 1 and auto[0]["post_match"]["post_person_nbr"] == "P1"
        assert auto[0]["input_officer"]["last_name"] == "Lunger"
        assert len(early) == 1 and "Common last name" in early[0]["review_reason"]
        assert len(failed) == 1 and failed[0]["review_reason"] == "No candidates found"

        import os
        assert os.path.exists(paths["csv"])
