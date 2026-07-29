"""Public provenance and component-fingerprint contracts for Task 10."""

from dataclasses import FrozenInstanceError, dataclass, replace

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
from pybacktest.application.provenance import (
    ProvenanceDescriptor,
    python_strategy_provenance,
)
from pybacktest.application.requests import BacktestRequest
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.errors import AdapterContractError, ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Quantity
from pybacktest.engine import BacktestEngine
from pybacktest.ports.components import ComponentDescriptor
from pybacktest.ports.strategy import StrategyContext
from pybacktest.risk import DefaultOrderSizer, LongShortRisk

from .test_engine import (
    _BuyWhenFlat,
    _engine,
    _simulation,
    _StaticSource,
    _two_bar_dataset,
)

_EXPLICIT_PROVENANCE = ProvenanceDescriptor(
    strategy_identity="mcp.compiled.momentum",
    strategy_fingerprint="a" * 64,
    spec_identity="strategyspec.momentum.v3",
    compiler_identity="pybacktest.mcp.compiler.v2",
    spec_fingerprint="b" * 64,
    schema_fingerprint="c" * 64,
)


def _base_broker_factory() -> SimulatedBrokerFactory:
    return SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(
            intrabar_policy=IntrabarPolicy.CONSERVATIVE
        ),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )


@dataclass(frozen=True)
class _StatelessStrategy:
    quantity: Quantity

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del context, market
        return ()


def test_provenance_descriptor_is_immutable_and_validated() -> None:
    with pytest.raises(FrozenInstanceError):
        _EXPLICIT_PROVENANCE.spec_identity = "changed"  # type: ignore[misc]
    with pytest.raises(ConfigurationError) as raised:
        replace(_EXPLICIT_PROVENANCE, strategy_fingerprint="not-a-digest")
    assert raised.value.code == "invalid_provenance_descriptor"
    with pytest.raises(ConfigurationError):
        replace(_EXPLICIT_PROVENANCE, spec_identity="  ")


def test_python_strategy_provenance_is_public_and_deterministic() -> None:
    plan = FeatureBuilder().plan()
    strategy = _StatelessStrategy(Quantity.of("1"))

    first = python_strategy_provenance(strategy, plan)
    second = python_strategy_provenance(strategy, plan)

    assert first == second
    assert first.strategy_identity.endswith("_StatelessStrategy")
    assert first.spec_identity == "python.strategy"
    assert first.compiler_identity == "pybacktest.session.python.v1"


def test_python_provenance_separates_identical_state_implementations() -> None:
    plan = FeatureBuilder().plan()

    stateless = python_strategy_provenance(
        _StatelessStrategy(Quantity.of("1")),
        plan,
    )
    buying = python_strategy_provenance(
        _BuyWhenFlat(Quantity.of("1")),
        plan,
    )

    assert stateless.strategy_fingerprint != buying.strategy_fingerprint


def _source_less_strategy() -> object:
    namespace: dict[str, object] = {}
    exec(
        compile(
            "class Generated:\n"
            "    def build_features(self, builder):\n"
            "        return builder.plan()\n"
            "    def on_bar(self, context, market):\n"
            "        return ()\n",
            "<generated>",
            "exec",
        ),
        namespace,
    )
    generated = namespace["Generated"]
    assert isinstance(generated, type)
    return generated()


def test_source_less_python_strategy_fails_closed() -> None:
    with pytest.raises(ConfigurationError) as raised:
        python_strategy_provenance(
            _source_less_strategy(),
            FeatureBuilder().plan(),
        )

    assert raised.value.code == "untrusted_strategy_provenance"


def test_explicit_descriptor_replaces_untrusted_python_provenance() -> None:
    dataset = _two_bar_dataset()
    request = BacktestRequest(
        strategy=_source_less_strategy(),
        simulation=_simulation(dataset),
        provenance=_EXPLICIT_PROVENANCE,
    )

    manifest = _engine(dataset).run(request).manifest

    assert manifest.strategy_identity == "mcp.compiled.momentum"
    assert manifest.strategy_fingerprint == "a" * 64
    assert manifest.spec_identity == "strategyspec.momentum.v3"
    assert manifest.compiler_identity == "pybacktest.mcp.compiler.v2"
    provenance = manifest.canonical_request["provenance"]
    assert provenance["spec_fingerprint"] == "b" * 64
    assert provenance["schema_fingerprint"] == "c" * 64


def test_create_session_binds_an_explicit_provenance_descriptor() -> None:
    dataset = _two_bar_dataset()
    session = _engine(dataset).create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
        run_id=RunId.parse("run_" + "9" * 32),
        provenance=_EXPLICIT_PROVENANCE,
    )

    assert session.provenance == _EXPLICIT_PROVENANCE
    observation = session.reset()
    step = session.advance((), observation=observation)
    assert step.observation is not None
    session.advance((), observation=step.observation)

    assert session.result().manifest.spec_identity == (
        "strategyspec.momentum.v3"
    )


class _CountingBrokerFactory:
    """A wrapper whose live call counter must not reach the fingerprint."""

    def __init__(self, inner: SimulatedBrokerFactory) -> None:
        self._inner = inner
        self.creates = 0

    def component_descriptor(self) -> ComponentDescriptor:
        return ComponentDescriptor(
            identity="tests.counting_broker_factory",
            version="1",
            configuration={"inner": type(self._inner).__qualname__},
        )

    def create(self, run_context):
        self.creates += 1
        return self._inner.create(run_context)


class _UndeclaredCountingBrokerFactory:
    def __init__(self, inner: SimulatedBrokerFactory) -> None:
        self._inner = inner
        self.creates = 0

    def create(self, run_context):
        self.creates += 1
        return self._inner.create(run_context)


def _engine_with(broker_factory) -> tuple[BacktestEngine, object]:
    dataset = _two_bar_dataset()
    return (
        BacktestEngine(
            data_source=_StaticSource(dataset),
            broker_factory=broker_factory,
            order_sizer=DefaultOrderSizer(),
            risk_policy=LongShortRisk(
                max_leverage=1,
                max_position_weight=None,
                allow_short=False,
            ),
        ),
        dataset,
    )


def test_declared_counting_factory_replays_to_one_fingerprint() -> None:
    factory = _CountingBrokerFactory(_base_broker_factory())
    engine, dataset = _engine_with(factory)
    request = BacktestRequest(
        strategy=_BuyWhenFlat(Quantity.of("1")),
        simulation=_simulation(dataset),
    )

    first = engine.run(request)
    second = engine.run(request)

    assert factory.creates == 2
    assert first.replay_fingerprint() == second.replay_fingerprint()


def test_undeclared_stateful_component_fails_closed() -> None:
    engine, dataset = _engine_with(
        _UndeclaredCountingBrokerFactory(_base_broker_factory())
    )

    with pytest.raises(ConfigurationError) as raised:
        engine.run(
            BacktestRequest(
                strategy=_BuyWhenFlat(Quantity.of("1")),
                simulation=_simulation(dataset),
            )
        )

    assert raised.value.code == "unsupported_component_state"


class _SharedBrokerFactory:
    def __init__(self, inner: SimulatedBrokerFactory) -> None:
        self._inner = inner
        self._broker = None

    def component_descriptor(self) -> ComponentDescriptor:
        return ComponentDescriptor(
            identity="tests.shared_broker_factory",
            version="1",
            configuration={},
        )

    def create(self, run_context):
        if self._broker is None:
            self._broker = self._inner.create(run_context)
        return self._broker


def test_broker_reused_across_sessions_is_rejected() -> None:
    engine, dataset = _engine_with(
        _SharedBrokerFactory(_base_broker_factory())
    )
    simulation = _simulation(dataset)
    plan = FeatureBuilder().plan()
    first = engine.create_session(
        simulation,
        feature_plan=plan,
        run_id=RunId.parse("run_" + "a" * 32),
    )
    second = engine.create_session(
        simulation,
        feature_plan=plan,
        run_id=RunId.parse("run_" + "b" * 32),
    )
    first.reset()

    with pytest.raises(AdapterContractError) as raised:
        second.reset()

    assert raised.value.code == "reused_broker_instance"
