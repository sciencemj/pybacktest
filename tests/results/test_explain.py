"""Causal trace tests use only explicitly recorded engine events."""

from datetime import UTC, datetime, timedelta

import pytest

from pybacktest.domain.identifiers import OrderId
from pybacktest.results.explain import build_trade_explanation
from pybacktest.results.models import (
    CausalStage,
    EngineEvent,
    EngineEventCode,
    UnknownTradeError,
)


def test_explanation_filters_one_order_and_preserves_recorded_order() -> None:
    first = OrderId.parse("order_" + "1" * 32)
    second = OrderId.parse("order_" + "2" * 32)
    observed_at = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    events = (
        EngineEvent(
            timestamp=observed_at,
            sequence=0,
            stage=CausalStage.of("feature"),
            code=EngineEventCode.of("decision.feature"),
            order_id=first,
            details={"values": {"fast": 101, "slow": 100}},
        ),
        EngineEvent(
            timestamp=observed_at,
            sequence=1,
            stage=CausalStage.of("feature"),
            code=EngineEventCode.of("decision.feature"),
            order_id=second,
            details={"values": {"fast": 99, "slow": 100}},
        ),
        EngineEvent(
            timestamp=observed_at + timedelta(seconds=1),
            sequence=2,
            stage=CausalStage.of("risk"),
            code=EngineEventCode.of("risk.approved"),
            order_id=first,
            details={"status": "approved"},
        ),
    )

    explanation = build_trade_explanation(events, first)

    assert explanation.order_id == first
    assert [entry.sequence for entry in explanation.entries] == [0, 2]
    assert [entry.stage.value for entry in explanation.entries] == [
        "feature",
        "risk",
    ]


def test_unknown_trade_is_a_typed_key_error() -> None:
    missing = OrderId.parse("order_" + "3" * 32)

    with pytest.raises(UnknownTradeError) as captured:
        build_trade_explanation((), missing)

    assert isinstance(captured.value, KeyError)
