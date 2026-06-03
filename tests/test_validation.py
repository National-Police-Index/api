"""Tests for agency validation = deterministic non-LE guard + LLM (injected fake)."""
from resolve.validation import validate_agency_match


def _yes(prompt, **kw):
    return "MATCH"


def _no(prompt, **kw):
    return "NO_MATCH"


def _boom(prompt, **kw):
    raise RuntimeError("llm down")


class TestGuardPaths:
    def test_empty_post_agency_rejected(self):
        ok, reason = validate_agency_match("Hayward PD", "", "", llm_fn=_yes)
        assert ok is False
        assert "empty" in reason.lower()

    def test_no_agencies_to_check_rejected(self):
        ok, reason = validate_agency_match("", "", "Hayward Police Department", llm_fn=_yes)
        assert ok is False

    def test_non_le_guard_blocks_da_to_pd_without_llm(self):
        # Guard must fire deterministically; _boom would raise if the LLM were called.
        ok, reason = validate_agency_match(
            "Alameda County District Attorney", "", "Hayward Police Department", llm_fn=_boom,
        )
        assert ok is False
        assert "non-le" in reason.lower()

    def test_guard_bypassed_when_le_agency_mentioned(self):
        ok, reason = validate_agency_match(
            "Alameda County District Attorney",
            "['Hayward Police Department']",
            "Hayward Police Department",
            llm_fn=_yes,
        )
        assert ok is True


class TestLLMPaths:
    def test_llm_match_validates(self):
        ok, reason = validate_agency_match(
            "Hayward PD", "", "Hayward Police Department", llm_fn=_yes,
        )
        assert ok is True
        assert reason == ""

    def test_llm_no_match_rejects(self):
        ok, reason = validate_agency_match(
            "Corona PD", "", "Riverside Police Department", llm_fn=_no,
        )
        assert ok is False
        assert "cannot be validated" in reason.lower()

    def test_llm_error_is_caught(self):
        ok, reason = validate_agency_match(
            "Hayward PD", "", "Hayward Police Department", llm_fn=_boom,
        )
        assert ok is False
        assert "error" in reason.lower()
