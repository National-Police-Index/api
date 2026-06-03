"""Tests for the resolve CLI: from-name (direct/API) and from-csv (batch)."""
import datetime
import os

from shared.models import OfficerMention
from resolve.cli import build_parser, cmd_from_name, cmd_from_csv
from resolve.pipeline import MentionResult


class FakeMatcher:
    def __init__(self):
        self.seen = []

    def resolve_one(self, mention):
        self.seen.append(mention)
        return MentionResult(mention, "auto_matched", "",
                             match={"post_person_nbr": "P1", "post_agency_name": "Hayward PD"})

    def resolve_batch(self, mentions):
        return [self.resolve_one(m) for m in mentions]


class TestParser:
    def test_from_name_parses(self):
        args = build_parser().parse_args([
            "from-name", "--first", "Scott", "--last", "Lunger",
            "--state", "CA", "--year", "2015", "--source-agency", "Hayward PD",
        ])
        assert args.command == "from-name"
        assert args.first == "Scott" and args.year == 2015

    def test_from_csv_parses(self):
        args = build_parser().parse_args([
            "from-csv", "--input", "x.csv", "--api", "http://localhost:8001",
            "--default-state", "CA",
        ])
        assert args.command == "from-csv"
        assert args.input == "x.csv" and args.default_state == "CA"


class TestFromName:
    def test_builds_mention_and_resolves(self):
        args = build_parser().parse_args([
            "from-name", "--first", "scott", "--last", "lunger",
            "--state", "CA", "--year", "2015", "--source-agency", "Hayward PD",
        ])
        matcher = FakeMatcher()
        result = cmd_from_name(args, matcher=matcher)
        assert isinstance(result, MentionResult)
        assert result.status == "auto_matched"
        m = matcher.seen[0]
        assert isinstance(m, OfficerMention)
        assert m.mention_first_name == "SCOTT"  # uppercased
        assert m.state == "CA"
        assert m.mention_incident_date == datetime.date(2015, 1, 1)


class TestFromCsv:
    def test_reads_csv_and_writes_outputs(self, tmp_path):
        csv = tmp_path / "mentions.csv"
        csv.write_text(
            "first_name,last_name,incident_year,source_agency,state\n"
            "Scott,Lunger,2015,Hayward PD,CA\n"
            "Jane,Doe,2018,Oakland PD,CA\n"
        )
        out_dir = tmp_path / "out"
        args = build_parser().parse_args([
            "from-csv", "--input", str(csv), "--output-dir", str(out_dir),
        ])
        matcher = FakeMatcher()
        paths = cmd_from_csv(args, matcher=matcher)
        assert len(matcher.seen) == 2
        assert os.path.exists(paths["auto_matched"])
