"""End-to-end incremental-short borrow-cost accounting contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from pybacktest.adapters.artifacts.local import LocalArtifactStore
from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    PerShareCommission,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
)
from pybacktest.application.requests import SimulationRequest
from pybacktest.data.calendar import CalendarPolicy
from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.data.features import FeatureBuilder
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    MarketOrderIntent,
    Order,
    OrderSide,
    TimeInForce,
)
from pybacktest.domain.portfolio import CashEventCode
from pybacktest.domain.time import DateRange, Timeframe
from pybacktest.engine.engine import BacktestEngine
from pybacktest.engine.session import Observation, SimulationSession
from pybacktest.results.metrics import MetricsConfig
from pybacktest.results.models import BacktestResult
from pybacktest.risk.policies import LongShortRisk
from tests.factories import instrument


class _StaticSource:
    def __init__(self, dataset: MarketDataSet) -> None:
        self.dataset = dataset

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        del universe, period, timeframe
        return self.dataset


@dataclass(frozen=True, slots=True)
class _PerShareBorrowCost:
    rate: Decimal = Decimal("1")

    calls: ClassVar[list[tuple[OrderSide, Decimal, Decimal]]] = []

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        type(self).calls.append((order.side, quantity.value, price.amount))
        return Money.of(quantity.value * self.rate, order.quote_currency)


@dataclass(frozen=True, slots=True)
class _InvalidOnSecondBorrowCost:
    kind: str

    calls: ClassVar[int] = 0

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        del quantity, price
        type(self).calls += 1
        if type(self).calls == 1:
            return Money.of("1", order.quote_currency)
        if self.kind == "wrong_type":
            return object()  # type: ignore[return-value]
        if self.kind == "wrong_currency":
            return Money.of("1", "EUR")
        if self.kind == "negative":
            return Money.of("-1", order.quote_currency)
        invalid = object.__new__(Money)
        object.__setattr__(invalid, "amount", Decimal("NaN"))
        object.__setattr__(invalid, "currency", order.quote_currency)
        return invalid


def _dataset(*, bar_count: int = 7, volume: float = 100.0) -> MarketDataSet:
    item = instrument()
    timestamps = np.datetime64("2024-01-02T14:30:00", "ns") + np.arange(
        bar_count,
        dtype="int64",
    ) * np.timedelta64(1, "D")
    prices = np.full(bar_count, 10.0)
    return MarketDataSet(
        series={
            item.id: BarSeries(
                timestamps=timestamps,
                open=prices,
                high=prices,
                low=prices,
                close=prices,
                volume=np.full(bar_count, volume),
            )
        },
        instruments={item.id: item},
        timeframe=Timeframe.days(1),
    )


def _session(
    *,
    borrow_cost: object,
    commission: object | None = None,
    liquidity: object | None = None,
    bar_count: int = 7,
    volume: float = 100.0,
    run_digit: str = "2",
) -> tuple[SimulationSession, InstrumentId]:
    dataset = _dataset(bar_count=bar_count, volume=volume)
    item_id = next(iter(dataset.instruments))
    engine = BacktestEngine(
        data_source=_StaticSource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(
                intrabar_policy=IntrabarPolicy.CONSERVATIVE,
            ),
            commission=commission or NoCommission(),
            slippage=NoSlippage(),
            liquidity=liquidity or NoLiquidityLimit(),
            borrow_cost=borrow_cost,  # type: ignore[arg-type]
        ),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("10"),
            max_position_weight=None,
            allow_short=True,
        ),
    )
    session = engine.create_session(
        SimulationRequest(
            universe=(item_id,),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("1000"),
            seed=17,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
        feature_plan=FeatureBuilder().plan(),
        run_id=RunId.parse("run_" + run_digit * 32),
    )
    return session, item_id


def _intent(
    instrument_id: InstrumentId,
    side: OrderSide,
    quantity: str,
) -> MarketOrderIntent:
    return MarketOrderIntent(
        instrument=instrument_id,
        side=side,
        quantity=Quantity.of(quantity),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("borrow_cost_test"),
    )


def _advance(
    session: SimulationSession,
    observation: Observation,
    *intents: MarketOrderIntent,
) -> Observation:
    step = session.advance(intents, observation=observation)
    assert step.observation is not None
    return step.observation


def _finish(
    session: SimulationSession,
    observation: Observation,
) -> BacktestResult:
    current = observation
    while not session.done:
        step = session.advance((), observation=current)
        if step.observation is not None:
            current = step.observation
    return session.result()


def _run_partial_short(*, run_digit: str = "2") -> BacktestResult:
    session, item_id = _session(
        borrow_cost=_PerShareBorrowCost(),
        commission=PerShareCommission(rate_per_share=Decimal("1")),
        liquidity=VolumeParticipationLimit(max_volume_ratio=Decimal("0.5")),
        bar_count=5,
        volume=4.0,
        run_digit=run_digit,
    )
    observation = session.reset()
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.SELL, "5"),
    )
    return _finish(session, observation)


def test_model_is_called_only_for_incremental_short_exposure() -> None:
    _PerShareBorrowCost.calls = []
    session, item_id = _session(borrow_cost=_PerShareBorrowCost())
    observation = session.reset()

    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.BUY, "10"),
    )
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.SELL, "4"),
    )
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.SELL, "10"),
    )
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.SELL, "3"),
    )
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.BUY, "5"),
    )
    result = _finish(session, observation)

    assert _PerShareBorrowCost.calls == [
        (OrderSide.SELL, Decimal("4"), Decimal("10.0")),
        (OrderSide.SELL, Decimal("3"), Decimal("10.0")),
    ]
    final = result.snapshots[-1]
    assert final.positions[item_id].quantity == Quantity.of("-2")
    assert final.cash == Money.usd("1013.0")
    assert final.equity == Money.usd("993.0")
    assert final.total_fees == Money.usd("0")
    assert [event.amount for event in final.cash_events] == [
        Money.usd("-4"),
        Money.usd("-3"),
    ]
    assert {event.code for event in final.cash_events} == {
        CashEventCode.BORROW_FEE,
    }
    assert all(fill.fee == Money.usd("0") for fill in result.fills)

    attributed = [
        event
        for event in result.events
        if event.code.value == "ledger.applied" and "cash_event_id" in event.details
    ]
    assert [event.details["cash_event_amount"] for event in attributed] == [
        "-4",
        "-3",
    ]
    assert [event.details["cash_event_code"] for event in attributed] == [
        "borrow_fee",
        "borrow_fee",
    ]


def test_partial_short_fills_charge_separately_from_commission() -> None:
    _PerShareBorrowCost.calls = []
    result = _run_partial_short()

    assert [fill.quantity for fill in result.fills] == [
        Quantity.of("2"),
        Quantity.of("2"),
        Quantity.of("1"),
    ]
    assert [fill.fee for fill in result.fills] == [
        Money.usd("2"),
        Money.usd("2"),
        Money.usd("1"),
    ]
    assert _PerShareBorrowCost.calls == [
        (OrderSide.SELL, Decimal("2"), Decimal("10.0")),
        (OrderSide.SELL, Decimal("2"), Decimal("10.0")),
        (OrderSide.SELL, Decimal("1"), Decimal("10.0")),
    ]
    final = result.snapshots[-1]
    assert final.cash == Money.usd("1040.0")
    assert final.equity == Money.usd("990.0")
    assert final.total_fees == Money.usd("5")
    assert [event.amount for event in final.cash_events] == [
        Money.usd("-2"),
        Money.usd("-2"),
        Money.usd("-1"),
    ]


def test_cash_event_identity_and_replay_are_deterministic() -> None:
    first = _run_partial_short(run_digit="2")
    repeated = _run_partial_short(run_digit="2")
    other_run = _run_partial_short(run_digit="3")

    first_ids = tuple(str(event.id) for event in first.snapshots[-1].cash_events)
    assert first_ids == (
        "cash_event_94baceb3d19151cb9bb2303177fb1ab4",
        "cash_event_a68e20ea33ac5d13bc77a6ec7ae7b0c1",
        "cash_event_d05acf0c92e15b028191282ee76c5e01",
    )
    assert (
        tuple(str(event.id) for event in repeated.snapshots[-1].cash_events)
        == first_ids
    )
    assert (
        tuple(str(event.id) for event in other_run.snapshots[-1].cash_events)
        != first_ids
    )
    assert first.replay_fingerprint() == repeated.replay_fingerprint()
    assert first.replay_fingerprint() == other_run.replay_fingerprint()


def test_borrow_fee_cash_events_persist_in_artifact_rows(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    result = _run_partial_short()
    assert [event.amount for event in result.snapshots[-1].cash_events] == [
        Money.usd("-2"),
        Money.usd("-2"),
        Money.usd("-1"),
    ]
    ref = LocalArtifactStore(tmp_path / "artifacts").write(result)

    equity_rows = pq.read_table(Path(ref.path) / "equity.parquet").to_pylist()
    persisted = json.loads(equity_rows[-1]["cash_events"])
    assert [row["amount"] for row in persisted] == ["-2", "-2", "-1"]
    assert [row["code"] for row in persisted] == [
        "borrow_fee",
        "borrow_fee",
        "borrow_fee",
    ]
    assert persisted == [
        {
            "cash_event_id": str(event.id),
            "timestamp": event.timestamp.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            "amount": str(event.amount.amount),
            "currency": "USD",
            "code": "borrow_fee",
        }
        for event in result.snapshots[-1].cash_events
    ]


def _engine_owned_state(session: SimulationSession) -> dict[str, object]:
    recorder = session._required_recorder()
    return {
        "cash_event_ordinal": getattr(session, "_cash_event_ordinal", 0),
        "fill_ordinal": session._fill_ordinal,
        "submitted_orders": dict(session._submitted_orders),
        "ledger": session._required_ledger().snapshot(),
        "recorder_market_timestamps": tuple(recorder._market_timestamps),
        "recorder_snapshots": tuple(recorder._snapshots),
        "recorder_orders": dict(recorder._orders),
        "recorder_fills": tuple(recorder._fills),
        "recorder_events": tuple(recorder._events),
    }


@pytest.mark.parametrize(
    "kind",
    ["wrong_type", "wrong_currency", "negative", "nonfinite"],
)
def test_invalid_second_borrow_cost_mutates_no_engine_owned_state(
    kind: str,
) -> None:
    _InvalidOnSecondBorrowCost.calls = 0
    session, item_id = _session(
        borrow_cost=_InvalidOnSecondBorrowCost(kind),
        liquidity=VolumeParticipationLimit(max_volume_ratio=Decimal("0.5")),
        bar_count=5,
        volume=4.0,
    )
    observation = session.reset()
    observation = _advance(
        session,
        observation,
        _intent(item_id, OrderSide.SELL, "5"),
    )
    before = _engine_owned_state(session)

    with pytest.raises(ConfigurationError, match="borrow cost"):
        session.advance((), observation=observation)

    assert _InvalidOnSecondBorrowCost.calls == 2
    assert _engine_owned_state(session) == before
