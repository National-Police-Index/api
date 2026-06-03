"""Stage-0 early-filter decision (route to manual review before expensive work).

Pure decision function: state requirement is a flag (all-states requires a state to
scope safely; CA `postie` can run unscoped). Same-name person count is supplied by the
caller so this stays decoupled from the API client.
"""
from resolve.pipeline import early_filter_decision

COMMON = {"SMITH", "JOHNSON"}


def _m(first="Scott", last="Lunger", state="CA"):
    return dict(mention_first_name=first, mention_last_name=last, state=state)


class TestEarlyFilter:
    def test_no_state_routed_when_state_required(self):
        skip, reason = early_filter_decision(_m(state=None), COMMON, same_name_count=0,
                                             require_state=True)
        assert skip is True
        assert "state" in reason.lower()

    def test_no_state_allowed_when_not_required(self):
        skip, reason = early_filter_decision(_m(state=None), COMMON, same_name_count=0,
                                             require_state=False)
        assert skip is False

    def test_common_last_name_routed(self):
        skip, reason = early_filter_decision(_m(last="Smith"), COMMON, same_name_count=0)
        assert skip is True
        assert "common" in reason.lower()

    def test_multiple_persons_routed(self):
        skip, reason = early_filter_decision(_m(), COMMON, same_name_count=2)
        assert skip is True
        assert "multiple" in reason.lower()

    def test_clean_mention_passes(self):
        skip, reason = early_filter_decision(_m(), COMMON, same_name_count=1)
        assert skip is False
        assert reason == ""
