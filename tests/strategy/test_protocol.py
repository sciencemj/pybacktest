from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal
from types import MappingProxyType

import numpy as np
import pytest

from pybacktest.data.features import FeatureBuilder, FeatureExecutor
from pybacktest.domain.errors import AdapterContractError, ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    OrderSide,
    TargetQuantity,
    TargetWeight,
    TimeInForce,
)
from pybacktest.ports.strategy import (
    PortfolioSnapshot,
    Strategy,
    StrategyContext,
    validate_strategy_output,
)
from pybacktest.strategy.intents import OrderIntent
from tests.factories import instrument as make_instrument
from tests.factories import one_instrument_dataset
from tests.factories import order as make_order


@dataclass(frozen=True)
class StaticPortfolio:
    positions: MappingProxyType[InstrumentId, Quantity]


def feature_context() -> StrategyContext:
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)
    active_orders = [make_order()]
    portfolio = StaticPortfolio(
        positions=MappingProxyType({instrument: Quantity.of("2")})
    )
    context = StrategyContext(
        timestamp=dataset.timestamps[1],
        portfolio=portfolio,
        active_orders=active_orders,
        features=features.view(dataset.timestamps[1]),
        run_id=RunId.parse(f"run_{1:032x}"),
    )
    active_orders.clear()
    return context


def test_strategy_context_is_typed_immutable_and_copies_active_orders():
    context = feature_context()

    assert len(context.active_orders) == 1
    assert isinstance(context.active_orders, tuple)
    assert context.features.at("close") == 2.0
    with pytest.raises(FrozenInstanceError):
        context.run_id = RunId.new()  # type: ignore[misc]


def test_strategy_protocol_exposes_feature_and_decision_boundaries():
    class NoopStrategy:
        def build_features(self, builder: FeatureBuilder):
            return builder.plan()

        def on_bar(self, context, market):
            del context, market
            return ()

    assert isinstance(NoopStrategy(), Strategy)


def test_validate_strategy_output_accepts_all_typed_intents_as_a_tuple():
    item = make_instrument()
    reason = DecisionReason.of("test_signal")
    active = make_order(item=item)
    intents: list[OrderIntent] = [
        TargetWeight(item.id, Decimal("0.5"), reason),
        TargetQuantity(item.id, Quantity.of("2"), reason),
        MarketOrderIntent(
            item.id,
            OrderSide.BUY,
            Quantity.of("3"),
            TimeInForce.DAY,
            reason,
        ),
        LimitOrderIntent(
            item.id,
            OrderSide.SELL,
            Quantity.of("1"),
            Money.usd("101"),
            TimeInForce.GOOD_TIL_CANCELLED,
            reason,
        ),
        CancelOrderIntent(active.id, reason),
    ]

    validated = validate_strategy_output(intents, universe={item.id})
    intents.clear()

    assert isinstance(validated, tuple)
    assert len(validated) == 5


@pytest.mark.parametrize(
    "value",
    [
        "not an intent sequence",
        b"not an intent sequence",
        [object()],
    ],
)
def test_validate_strategy_output_rejects_misleading_or_non_intents(
    value: object,
):
    item = make_instrument()

    with pytest.raises(AdapterContractError, match="intent"):
        validate_strategy_output(value, universe={item.id})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "intent",
    [
        TargetWeight(
            InstrumentId.parse("XNYS:IBM"),
            Decimal("0.1"),
            DecisionReason.of("foreign"),
        ),
        TargetQuantity(
            InstrumentId.parse("XNYS:IBM"),
            Quantity.of("1"),
            DecisionReason.of("foreign"),
        ),
        MarketOrderIntent(
            InstrumentId.parse("XNYS:IBM"),
            OrderSide.BUY,
            Quantity.of("1"),
            TimeInForce.DAY,
            DecisionReason.of("foreign"),
        ),
        LimitOrderIntent(
            InstrumentId.parse("XNYS:IBM"),
            OrderSide.BUY,
            Quantity.of("1"),
            Money.usd("10"),
            TimeInForce.DAY,
            DecisionReason.of("foreign"),
        ),
    ],
)
def test_validate_strategy_output_rejects_out_of_universe_instruments(
    intent: OrderIntent,
):
    with pytest.raises(AdapterContractError, match="universe"):
        validate_strategy_output(
            [intent],
            universe={make_instrument().id},
        )


def test_validate_strategy_output_rejects_duplicate_cancel_requests():
    active = make_order()
    reason = DecisionReason.of("cancel")

    with pytest.raises(AdapterContractError, match="duplicate cancel"):
        validate_strategy_output(
            [
                CancelOrderIntent(active.id, reason),
                CancelOrderIntent(active.id, reason),
            ],
            universe={active.instrument},
        )


def test_strategy_context_rejects_feature_view_timestamp_mismatch():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)

    with pytest.raises(ConfigurationError, match=r"feature.*timestamp"):
        StrategyContext(
            timestamp=dataset.timestamps[1],
            portfolio=StaticPortfolio(
                positions=MappingProxyType({instrument: Quantity.of("1")})
            ),
            active_orders=(),
            features=features.view(dataset.timestamps[0]),
            run_id=RunId.parse(f"run_{2:032x}"),
        )


def test_portfolio_snapshot_defensively_copies_positions():
    item = make_instrument()
    source_positions = {item.id: Quantity.of("2")}

    snapshot = PortfolioSnapshot(positions=source_positions)
    source_positions[item.id] = Quantity.of("99")

    assert isinstance(snapshot.positions, MappingProxyType)
    assert snapshot.positions[item.id] == Quantity.of("2")
    with pytest.raises(TypeError):
        snapshot.positions[item.id] = Quantity.of("3")  # type: ignore[index]


def test_strategy_context_replaces_structural_portfolio_with_frozen_snapshot():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)
    source_positions = {instrument: Quantity.of("2")}
    source = StaticPortfolio(positions=source_positions)  # type: ignore[arg-type]

    context = StrategyContext(
        timestamp=dataset.timestamps[1],
        portfolio=source,
        active_orders=(),
        features=features.view(dataset.timestamps[1]),
        run_id=RunId.parse(f"run_{3:032x}"),
    )
    source_positions[instrument] = Quantity.of("99")

    assert type(context.portfolio) is PortfolioSnapshot
    assert context.portfolio.positions[instrument] == Quantity.of("2")


@pytest.mark.parametrize(
    "positions",
    [
        [],
        {"XNAS:AAPL": Quantity.of("1")},
        {make_instrument().id: "not a quantity"},
    ],
)
def test_portfolio_snapshot_rejects_invalid_position_mappings(
    positions: object,
):
    with pytest.raises(ConfigurationError, match="positions"):
        PortfolioSnapshot(positions=positions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "not_a_scalar",
    [
        np.array([np.datetime64("2024-01-03", "ns")]),
        np.array(np.datetime64("2024-01-03", "ns")),
    ],
)
def test_strategy_context_rejects_datetime_arrays(
    not_a_scalar: object,
):
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)

    with pytest.raises(ConfigurationError, match="scalar"):
        StrategyContext(
            timestamp=not_a_scalar,  # type: ignore[arg-type]
            portfolio=StaticPortfolio(
                positions=MappingProxyType({instrument: Quantity.of("1")})
            ),
            active_orders=(),
            features=features.view(dataset.timestamps[1]),
            run_id=RunId.parse(f"run_{4:032x}"),
        )
