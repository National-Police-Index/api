"""Tests for candidate filtering, including the OPTIONAL county / agency_type logic.

The consolidated pipeline must preserve match.py's rich CA behavior (county filter,
agency_type mask) but apply each only when the data supports it:
  - county filter: applied iff a source county is known AND agency_type != CORRECTIONS
  - agency_type mask: applied iff candidate data carries real (non-uniform-POLICE) types
  - date + name filters: always applied
"""
import pandas as pd

from resolve.candidates import (
    in_date_range,
    filter_by_name,
    filter_by_county,
    has_real_agency_type,
    filter_by_agency_type,
    select_candidates,
)


def _post(**overrides):
    """One POST employment record row with sensible defaults."""
    base = dict(
        post_person_nbr="P1",
        post_first_name="Scott",
        post_last_name="Lunger",
        post_agency_name="Hayward Police Department",
        post_agency_type="POLICE",
        post_start_date="2010-01-01",
        post_end_date="2020-01-01",
        county="Alameda",
    )
    base.update(overrides)
    return base


class TestDateRange:
    def test_in_range(self):
        df = pd.DataFrame([_post(post_start_date="2010-01-01", post_end_date="2020-01-01")])
        assert in_date_range(df, incident_year=2015).iloc[0]

    def test_out_of_range_before(self):
        df = pd.DataFrame([_post(post_start_date="2018-01-01", post_end_date="2020-01-01")])
        assert not in_date_range(df, incident_year=2015).iloc[0]

    def test_empty_end_date_treated_as_current(self):
        df = pd.DataFrame([_post(post_start_date="2010-01-01", post_end_date="")])
        assert in_date_range(df, incident_year=2015).iloc[0]

    def test_handles_tz_aware_iso(self):
        df = pd.DataFrame([_post(post_start_date="2010-01-01T00:00:00Z",
                                 post_end_date="2020-01-01T00:00:00Z")])
        assert in_date_range(df, incident_year=2015).iloc[0]


class TestNameNet:
    def test_exact_first_and_last(self):
        df = pd.DataFrame([_post(post_first_name="Scott", post_last_name="Lunger")])
        out = filter_by_name(df, first_name="Scott", last_name="Lunger")
        assert len(out) == 1

    def test_first_prefix_with_exact_last(self):
        # "Scotty" shares 2-char prefix "Sc" with "Scott"; last name exact
        df = pd.DataFrame([_post(post_first_name="Scotty", post_last_name="Lunger")])
        out = filter_by_name(df, first_name="Scott", last_name="Lunger")
        assert len(out) == 1

    def test_case_insensitive(self):
        df = pd.DataFrame([_post(post_first_name="SCOTT", post_last_name="LUNGER")])
        out = filter_by_name(df, first_name="scott", last_name="lunger")
        assert len(out) == 1

    def test_different_name_excluded(self):
        df = pd.DataFrame([_post(post_first_name="Michael", post_last_name="Brown")])
        out = filter_by_name(df, first_name="Scott", last_name="Lunger")
        assert len(out) == 0


class TestCountyFilter:
    def test_keeps_person_with_any_record_in_county(self):
        df = pd.DataFrame([
            _post(post_person_nbr="P1", county="Alameda"),
            _post(post_person_nbr="P1", county="Contra Costa"),  # same person, other county
            _post(post_person_nbr="P2", county="Riverside"),
        ])
        out = filter_by_county(df, source_county="Alameda")
        assert set(out["post_person_nbr"]) == {"P1"}  # both P1 rows kept, P2 dropped

    def test_drops_all_when_no_match(self):
        df = pd.DataFrame([_post(post_person_nbr="P2", county="Riverside")])
        out = filter_by_county(df, source_county="Alameda")
        assert len(out) == 0


class TestAgencyTypeDetection:
    def test_uniform_police_is_not_real(self):
        df = pd.DataFrame([_post(post_agency_type="POLICE"), _post(post_agency_type="POLICE")])
        assert has_real_agency_type(df) is False

    def test_mixed_types_is_real(self):
        df = pd.DataFrame([_post(post_agency_type="POLICE"),
                           _post(post_agency_type="CORRECTIONS")])
        assert has_real_agency_type(df) is True

    def test_agency_type_mask_keeps_matching(self):
        df = pd.DataFrame([_post(post_agency_type="CORRECTIONS"),
                           _post(post_agency_type="POLICE")])
        out = filter_by_agency_type(df, agency_type="CORRECTIONS")
        assert set(out["post_agency_type"]) == {"CORRECTIONS"}


class TestSelectCandidatesOptionalLogic:
    """The orchestrator: same code, two data regimes."""

    def test_all_states_regime_no_county_no_types(self):
        # all-states: county is None for every row, agency_type uniformly POLICE.
        df = pd.DataFrame([
            _post(post_person_nbr="P1", county=None, post_agency_type="POLICE"),
            _post(post_person_nbr="P2", post_first_name="Other", post_last_name="Person",
                  county=None, post_agency_type="POLICE"),
        ])
        out = select_candidates(
            df, first_name="Scott", last_name="Lunger", incident_year=2015,
            agency_type="POLICE", source_county=None,
        )
        # name+date keep P1, drop the unrelated P2; county/type filters are no-ops
        assert set(out["post_person_nbr"]) == {"P1"}

    def test_ca_regime_applies_county_filter(self):
        df = pd.DataFrame([
            _post(post_person_nbr="P1", county="Alameda"),
            _post(post_person_nbr="P2", county="Riverside"),  # same name, wrong county
        ])
        out = select_candidates(
            df, first_name="Scott", last_name="Lunger", incident_year=2015,
            agency_type="POLICE", source_county="Alameda",
        )
        assert set(out["post_person_nbr"]) == {"P1"}

    def test_corrections_skips_county_filter(self):
        # CORRECTIONS officers work statewide -> county filter must be skipped
        df = pd.DataFrame([
            _post(post_person_nbr="P1", county="Riverside", post_agency_type="CORRECTIONS"),
        ])
        out = select_candidates(
            df, first_name="Scott", last_name="Lunger", incident_year=2015,
            agency_type="CORRECTIONS", source_county="Alameda",
        )
        assert set(out["post_person_nbr"]) == {"P1"}
