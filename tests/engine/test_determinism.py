from dataclasses import dataclass
from decimal import Decimal

import pytest

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
)
from pybacktest.application.requests import BacktestRequest
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    MarketOrderIntent,
    OrderSide,
    TimeInForce,
)
from pybacktest.engine import BacktestEngine
from pybacktest.ports.strategy import StrategyContext
from pybacktest.risk import DefaultOrderSizer, LongShortRisk

from .test_engine import (
    _BuyWhenFlat,
    _engine,
    _simulation,
    _StaticSource,
    _two_bar_dataset,
)


def _normalized_orders(result):
    return tuple(
        (
            order.instrument,
            order.side,
            order.type,
            order.quantity,
            order.quote_currency,
            order.limit_price,
            order.time_in_force,
            order.submitted_at,
            order.active_from,
            order.reason,
            order.status,
            order.filled_quantity,
        )
        for order in result.orders
    )


def _normalized_fills(result):
    return tuple(
        (
            fill.instrument,
            fill.side,
            fill.quantity,
            fill.price,
            fill.fee,
            fill.timestamp,
        )
        for fill in result.fills
    )


def test_generated_run_ids_differ_but_replay_outputs_are_equal() -> None:
    dataset = _two_bar_dataset()
    request = BacktestRequest(
        strategy=_BuyWhenFlat(Quantity.of("1")),
        simulation=_simulation(dataset),
    )
    engine = _engine(dataset)

    first = engine.run(request)
    second = engine.run(request)

    assert first.run_id != second.run_id
    assert first.replay_fingerprint() == second.replay_fingerprint()
    assert _normalized_orders(first) == _normalized_orders(second)
    assert _normalized_fills(first) == _normalized_fills(second)
    assert first.snapshots == second.snapshots
    assert first.summary == second.summary
    assert first.manifest.strategy_fingerprint == second.manifest.strategy_fingerprint
    assert first.manifest.dataset_fingerprint == second.manifest.dataset_fingerprint


@dataclass(frozen=True)
class _UnsupportedStateStrategy:
    dependency: object

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del context, market
        return ()


def test_unsupported_strategy_state_fails_with_strategy_specific_code() -> None:
    dataset = _two_bar_dataset()
    request = BacktestRequest(
        strategy=_UnsupportedStateStrategy(object()),
        simulation=_simulation(dataset),
    )

    with pytest.raises(ConfigurationError) as raised:
        _engine(dataset).run(request)

    assert raised.value.code == "unsupported_strategy_state"
    assert "strategy" in str(raised.value)


def test_interleaved_sessions_do_not_share_execution_or_ledger_state() -> None:
    dataset = _two_bar_dataset()
    engine = _engine(dataset)
    simulation = _simulation(dataset)
    plan = FeatureBuilder().plan()
    first_session = engine.create_session(
        simulation,
        feature_plan=plan,
        run_id=RunId.parse("run_" + "3" * 32),
    )
    second_session = engine.create_session(
        simulation,
        feature_plan=plan,
        run_id=RunId.parse("run_" + "4" * 32),
    )
    first_observation = first_session.reset()
    second_observation = second_session.reset()
    instrument_id = next(iter(first_observation.market.bars))
    buy = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("isolated_buy"),
    )

    first_step = first_session.advance(
        (buy,),
        observation=first_observation,
    )

    assert first_step.observation is not None
    assert first_step.observation.portfolio.positions[
        instrument_id
    ].quantity == Quantity.of("1")
    assert second_observation.active_orders == ()
    assert second_observation.portfolio.positions == {}

    second_step = second_session.advance(
        (),
        observation=second_observation,
    )
    assert second_step.observation is not None
    assert second_step.observation.portfolio.positions == {}
    first_session.advance((), observation=first_step.observation)
    second_session.advance((), observation=second_step.observation)
    first_result = first_session.result()
    second_result = second_session.result()

    assert len(first_result.fills) == 1
    assert second_result.fills == ()
    assert {order.id for order in first_result.orders}.isdisjoint(
        order.id for order in second_result.orders
    )


def test_local_adapter_identity_is_stable_manifest_metadata() -> None:
    dataset = _two_bar_dataset()

    class LocalSource(_StaticSource):
        pass

    engine = BacktestEngine(
        data_source=LocalSource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=NoCommission(),
            slippage=NoSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        order_sizer=DefaultOrderSizer(),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )

    result = engine.run(
        BacktestRequest(
            strategy=_BuyWhenFlat(Quantity.of("1")),
            simulation=_simulation(dataset),
        )
    )

    identity = result.manifest.adapter_versions["data"]
    assert "LocalSource" in identity
    assert "<" not in identity
    assert ">" not in identity


@dataclass(frozen=True)
class _RandomTickSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price: Money,
        market,
        rng,
    ) -> Money:
        del order, quantity, market
        cents = int(rng.integers(0, 100))
        return Money.of(
            reference_price.amount + Decimal(cents) / Decimal("100"),
            reference_price.currency,
        )


def test_interleaved_sessions_own_independent_seeded_rng_streams() -> None:
    dataset = _two_bar_dataset()
    engine = BacktestEngine(
        data_source=_StaticSource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=NoCommission(),
            slippage=_RandomTickSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        order_sizer=DefaultOrderSizer(),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )
    plan = FeatureBuilder().plan()
    first = engine.create_session(
        _simulation(dataset),
        feature_plan=plan,
        run_id=RunId.parse("run_" + "6" * 32),
    )
    second = engine.create_session(
        _simulation(dataset),
        feature_plan=plan,
        run_id=RunId.parse("run_" + "7" * 32),
    )
    first_observation = first.reset()
    second_observation = second.reset()
    instrument_id = next(iter(first_observation.market.bars))
    buy = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("rng_isolation"),
    )

    first_step = first.advance((buy,), observation=first_observation)
    second_step = second.advance((buy,), observation=second_observation)
    assert first_step.observation is not None
    assert second_step.observation is not None
    first.advance((), observation=first_step.observation)
    second.advance((), observation=second_step.observation)

    assert first.result().fills[0].price == Money.usd("105.13")
    assert second.result().fills[0].price == Money.usd("105.13")


def test_external_action_session_has_explicit_deterministic_provenance() -> None:
    dataset = _two_bar_dataset()
    session = _engine(dataset).create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
        run_id=RunId.parse("run_" + "8" * 32),
    )
    first = session.reset()
    step = session.advance((), observation=first)
    assert step.observation is not None
    session.advance((), observation=step.observation)

    manifest = session.result().manifest

    assert manifest.strategy_identity == "external.actions"
    assert manifest.spec_identity == "external.actions"
    assert manifest.compiler_identity == "pybacktest.session.external.v1"
    assert manifest.strategy_fingerprint != "0" * 64
