"""Tests for input prep: officer-uid hashing and incident-year derivation.

Ported behavior from legacy resolve/src/match*.py (generate_officer_uid) and
resolve/src/helpers.py (ensure_incident_year_column).
"""
import pandas as pd
import pytest

from resolve.io import generate_officer_uid, ensure_incident_year_column


class TestGenerateOfficerUID:
    def test_deterministic_for_same_inputs(self):
        row = pd.Series({
            "first_name": "Scott", "last_name": "Lunger",
            "provisional_case_name": "case-1", "incident_year": "2015",
            "incident_month": "", "incident_date": "", "source_agency": "Hayward PD",
        })
        assert generate_officer_uid(row) == generate_officer_uid(row.copy())

    def test_is_sha256_hex(self):
        row = pd.Series({"first_name": "A", "last_name": "B"})
        uid = generate_officer_uid(row)
        assert len(uid) == 64
        assert all(c in "0123456789abcdef" for c in uid)

    def test_differs_when_name_differs(self):
        base = {"first_name": "Scott", "last_name": "Lunger", "source_agency": "Hayward PD"}
        a = generate_officer_uid(pd.Series(base))
        b = generate_officer_uid(pd.Series({**base, "last_name": "Landreth"}))
        assert a != b

    def test_tolerates_missing_fields(self):
        # No KeyError even with a near-empty row
        generate_officer_uid(pd.Series({"first_name": "X"}))


class TestEnsureIncidentYearColumn:
    def test_returns_unchanged_when_year_present(self):
        df = pd.DataFrame({"incident_year": [2015, 2018]})
        out = ensure_incident_year_column(df)
        assert list(out["incident_year"]) == [2015, 2018]

    def test_derives_year_from_incident_date(self):
        df = pd.DataFrame({"incident_date": ["2015-06-01", "2018-01-15"]})
        out = ensure_incident_year_column(df)
        assert list(out["incident_year"]) == [2015, 2018]

    def test_picks_most_recent_when_comma_separated(self):
        df = pd.DataFrame({"incident_date": ["2010-01-01, 2019-05-05, 2015-03-03"]})
        out = ensure_incident_year_column(df)
        assert out["incident_year"].iloc[0] == 2019

    def test_raises_when_neither_column_present(self):
        with pytest.raises(ValueError):
            ensure_incident_year_column(pd.DataFrame({"name": ["x"]}))
