"""Public provenance and component-fingerprint contracts for Task 10."""

import inspect
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from pybacktest._introspection import deterministic_instance_state
from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
)
from pybacktest.application import provenance as provenance_module
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
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
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


class _SlottedConfigBase:
    __slots__ = ("threshold",)


class _SlottedConfigStrategy(_SlottedConfigBase):
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del context, market
        return ()


def test_inherited_slotted_configuration_changes_the_fingerprint() -> None:
    plan = FeatureBuilder().plan()

    low = python_strategy_provenance(_SlottedConfigStrategy(1), plan)
    high = python_strategy_provenance(_SlottedConfigStrategy(999), plan)

    assert low.strategy_identity == high.strategy_identity
    assert low.strategy_fingerprint != high.strategy_fingerprint


def test_strategy_fingerprint_depends_on_the_implementation_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _StatelessStrategy(Quantity.of("1"))
    plan = FeatureBuilder().plan()
    baseline = python_strategy_provenance(strategy, plan)

    monkeypatch.setattr(
        provenance_module,
        "_implementation_digest",
        lambda strategy_type: "0" * 64,
    )
    mutated = python_strategy_provenance(strategy, plan)

    assert mutated.strategy_identity == baseline.strategy_identity
    assert mutated.strategy_fingerprint != baseline.strategy_fingerprint


class _CounterBase:
    __slots__ = ("creates",)


class _InheritedCounterBrokerFactory(_CounterBase):
    """Its only instance state is a counter held in an inherited slot."""

    __slots__ = ()

    def __init__(self) -> None:
        self.creates = 0

    def create(self, run_context):
        self.creates += 1
        return _base_broker_factory().create(run_context)


def test_inherited_mutable_counter_fails_the_component_gate() -> None:
    engine, dataset = _engine_with(_InheritedCounterBrokerFactory())

    with pytest.raises(ConfigurationError) as raised:
        engine.run(
            BacktestRequest(
                strategy=_BuyWhenFlat(Quantity.of("1")),
                simulation=_simulation(dataset),
            )
        )

    assert raised.value.code == "unsupported_component_state"


class _DuplicateSlotBase:
    __slots__ = ("threshold",)

    def __init__(self, base_threshold: int) -> None:
        _DuplicateSlotBase.threshold.__set__(self, base_threshold)


class _DuplicateSlotStrategy(_DuplicateSlotBase):
    """Re-declares a real slot name, so two distinct descriptors exist."""

    __slots__ = ("threshold",)

    def __init__(self, base_threshold: int, own_threshold: int) -> None:
        super().__init__(base_threshold)
        self.threshold = own_threshold

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del context, market
        return ()


def test_duplicate_mro_slot_names_are_handled_deterministically() -> None:
    plan = FeatureBuilder().plan()

    try:
        low = python_strategy_provenance(
            _DuplicateSlotStrategy(1, 7),
            plan,
        )
        high = python_strategy_provenance(
            _DuplicateSlotStrategy(999, 7),
            plan,
        )
    except ConfigurationError as error:
        assert error.code == "duplicate_slot_name"
        return

    assert low.strategy_fingerprint != high.strategy_fingerprint


class _DuplicateSlotFactoryBase:
    __slots__ = ("creates",)

    def __init__(self, base_creates: int) -> None:
        _DuplicateSlotFactoryBase.creates.__set__(self, base_creates)


class _DuplicateSlotBrokerFactory(_DuplicateSlotFactoryBase):
    __slots__ = ("creates",)

    def __init__(self, base_creates: int) -> None:
        super().__init__(base_creates)
        self.creates = 0

    def create(self, run_context):
        self.creates += 1
        return _base_broker_factory().create(run_context)


def test_duplicate_mro_slot_component_state_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        deterministic_instance_state(_DuplicateSlotBrokerFactory(1))
    assert raised.value.code == "duplicate_slot_name"

    engine, dataset = _engine_with(_DuplicateSlotBrokerFactory(1))
    with pytest.raises(ConfigurationError) as run_raised:
        engine.run(
            BacktestRequest(
                strategy=_BuyWhenFlat(Quantity.of("1")),
                simulation=_simulation(dataset),
            )
        )
    assert run_raised.value.code == "duplicate_slot_name"


def test_ordinary_inherited_slots_still_resolve_without_rejection() -> None:
    state = deterministic_instance_state(_SlottedConfigStrategy(11))

    assert state == {"threshold": 11}


class _InheritedBehaviorBase:
    """Holds the behaviour; the concrete strategy adds nothing of its own."""

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del context, market
        return ()


class _InheritsBehavior(_InheritedBehaviorBase):
    pass


def _entry_for(descriptor, qualname: str) -> dict[str, str] | None:
    for entry in descriptor:
        if entry["qualname"] == qualname:
            return entry
    return None


def _leaf_entry(descriptor) -> dict[str, str]:
    entry = _entry_for(descriptor, _InheritsBehavior.__qualname__)
    assert entry is not None, "the leaf class must always be described"
    return entry


def test_inherited_behavior_source_reaches_the_strategy_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an inherited base's source changes; the fingerprint must follow.

    Every other fingerprint input is held constant and asserted so: identity,
    configuration, package identity, feature plan, and the leaf class's own
    source. The counterfactual below pins that a leaf-only digest genuinely
    would collide, so this test cannot pass for any reason other than the
    inherited behavioural source reaching the fingerprint.
    """
    plan = FeatureBuilder().plan()
    strategy = _InheritsBehavior()
    baseline = python_strategy_provenance(strategy, plan)
    baseline_descriptor = provenance_module._implementation_descriptor(
        _InheritsBehavior
    )
    baseline_package = provenance_module._package_identity(_InheritsBehavior)
    baseline_state = deterministic_instance_state(strategy)
    baseline_leaf_source = inspect.getsource(_InheritsBehavior)

    real_getsource = inspect.getsource

    def patched(obj):
        if obj is _InheritedBehaviorBase:
            return real_getsource(obj) + "\n# inherited behaviour changed\n"
        return real_getsource(obj)

    monkeypatch.setattr(provenance_module.inspect, "getsource", patched)
    mutated = python_strategy_provenance(strategy, plan)
    mutated_descriptor = provenance_module._implementation_descriptor(_InheritsBehavior)

    # Everything except the inherited base source is provably unchanged.
    assert mutated.strategy_identity == baseline.strategy_identity
    assert provenance_module._package_identity(_InheritsBehavior) == baseline_package
    assert deterministic_instance_state(strategy) == baseline_state
    # The patch rewrites the base only; the leaf class source is untouched.
    assert (
        provenance_module.inspect.getsource(_InheritsBehavior) == baseline_leaf_source
    )

    # The counterfactual: a digest built from the leaf class alone is
    # byte-identical across the change, so leaf-only fingerprinting would
    # collide here. This is what makes the assertion below meaningful.
    assert _leaf_entry(mutated_descriptor) == _leaf_entry(baseline_descriptor)
    assert provenance_module._digest(
        (_leaf_entry(mutated_descriptor),)
    ) == provenance_module._digest((_leaf_entry(baseline_descriptor),))

    # Only the inherited base entry moved, and the fingerprint moved with it.
    base_qualname = _InheritedBehaviorBase.__qualname__
    baseline_base = _entry_for(baseline_descriptor, base_qualname)
    mutated_base = _entry_for(mutated_descriptor, base_qualname)
    assert baseline_base is not None, "base must be in the MRO descriptor"
    assert mutated_base is not None, "base must be in the MRO descriptor"
    assert mutated_base != baseline_base
    assert mutated.strategy_fingerprint != baseline.strategy_fingerprint


def test_implementation_descriptor_excludes_framework_internals() -> None:
    descriptor = provenance_module._implementation_descriptor(_InheritsBehavior)

    qualnames = [entry["qualname"] for entry in descriptor]
    assert "_InheritsBehavior" in qualnames
    assert "_InheritedBehaviorBase" in qualnames
    assert "object" not in qualnames
    assert all(not entry["module"].startswith("pybacktest.") for entry in descriptor)


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

    assert session.result().manifest.spec_identity == ("strategyspec.momentum.v3")


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
    engine, dataset = _engine_with(_SharedBrokerFactory(_base_broker_factory()))
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
