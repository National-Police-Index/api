"""Characterization + unit tests for the deterministic agency guard.

This logic is the precision backbone that replaces county/agency_type filtering.
It must be importable WITHOUT an OpenAI key (no LLM client at import time).
Ported behavior from the legacy resolve/src/helpers.py.
"""
from resolve.agency import (
    is_non_le_agency,
    is_le_agency,
    all_non_le,
    parse_agencies_to_check,
    non_le_guard,
)


class TestNonLEClassification:
    def test_district_attorney_is_non_le(self):
        assert is_non_le_agency("Alameda County District Attorney") is True

    def test_coroner_is_non_le(self):
        assert is_non_le_agency("Los Angeles County Coroner") is True

    def test_bare_da_token_is_non_le(self):
        # " da " padded-token match
        assert is_non_le_agency("Alameda County DA") is True

    def test_police_department_is_not_non_le(self):
        assert is_non_le_agency("Hayward Police Department") is False

    def test_mixed_string_with_le_keyword_is_le(self):
        # "Sheriff's Office / DA" contains an LE keyword -> treated as LE, not non-LE
        assert is_non_le_agency("Sheriff's Office / DA") is False

    def test_empty_is_not_non_le(self):
        assert is_non_le_agency("") is False


class TestLEClassification:
    def test_police_is_le(self):
        assert is_le_agency("Hayward Police Department") is True

    def test_sheriff_is_le(self):
        assert is_le_agency("Napa County Sheriff's Office") is True

    def test_da_is_not_le(self):
        assert is_le_agency("Alameda County DA") is False

    def test_empty_is_not_le(self):
        assert is_le_agency("") is False


class TestAllNonLE:
    def test_all_non_le_true_when_every_agency_non_le(self):
        assert all_non_le(["Alameda County DA", "County Coroner"]) is True

    def test_all_non_le_false_when_any_le_present(self):
        assert all_non_le(["Alameda County DA", "Oakland Police Department"]) is False

    def test_all_non_le_false_when_empty(self):
        assert all_non_le([]) is False


class TestParseAgenciesToCheck:
    def test_includes_mention_agency_and_list(self):
        agencies = parse_agencies_to_check(
            mention_agency="Hayward Police Department",
            mentioned_agencies="['Oakland Police Department', 'CHP']",
        )
        assert "Hayward Police Department" in agencies
        assert "Oakland Police Department" in agencies
        assert "CHP" in agencies

    def test_handles_empty_mentioned(self):
        agencies = parse_agencies_to_check("Hayward PD", "")
        assert agencies == ["Hayward PD"]

    def test_non_list_string_treated_as_single(self):
        agencies = parse_agencies_to_check("", "Some Agency Name")
        assert "Some Agency Name" in agencies


class TestNonLEGuard:
    def test_blocks_da_source_to_le_post(self):
        blocked, reason = non_le_guard(
            ["Alameda County District Attorney"], "Hayward Police Department"
        )
        assert blocked is True
        assert reason  # non-empty explanation

    def test_bypassed_when_le_agency_present(self):
        blocked, reason = non_le_guard(
            ["Alameda County DA", "Oakland Police Department"], "Hayward Police Department"
        )
        assert blocked is False

    def test_not_blocked_when_post_agency_non_le(self):
        blocked, reason = non_le_guard(
            ["Alameda County DA"], "Alameda County District Attorney"
        )
        assert blocked is False
