"""Cross-field invariants for typed calendar policy construction."""

import pytest

from pybacktest.data.calendar import CalendarMode, CalendarPolicy
from pybacktest.domain.errors import ConfigurationError


def test_intersection_policy_rejects_nonzero_staleness_directly() -> None:
    with pytest.raises(
        ConfigurationError,
        match="intersection calendars require max_staleness_bars=0",
    ):
        CalendarPolicy(
            mode=CalendarMode.INTERSECTION,
            max_staleness_bars=1,
        )
