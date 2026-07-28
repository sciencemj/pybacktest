# Pybacktest V2 StrategySpec and Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned, safe, machine-readable `StrategySpec` that compiles only allowlisted components into the V2 `Strategy` protocol, produces identical semantics to handwritten Python strategies, composes stricter strategy risk with engine risk, and records reproducible AI experiment lineage.

**Architecture:** Pydantic discriminated unions validate JSON shape; a separate semantic validator resolves instruments, feature references, graph dependencies, units, and policy limits. A frozen component registry maps stable public component names to typed compiler factories without accepting import paths or executable expressions. The compiler emits a `CompiledStrategyBundle` consumed by `BacktestService.run_spec()`, while a local SQLite experiment store records specs, feedback, run IDs, metrics, and lineage.

**Tech Stack:** Python 3.11+, Pybacktest V2 core interfaces, Pydantic 2, stdlib hashlib/json/sqlite3, pytest, Hypothesis, Ruff, ty, uv.

## Global Constraints

- Complete and verify `2026-07-29-pybacktest-v2-core.md` first.
- `StrategySpec.spec_version` is exactly `"1"` for this release.
- Every Pydantic model uses `ConfigDict(extra="forbid", frozen=True)`.
- JSON may contain only discriminated typed fields; no Python code, callable, import path, URL, filesystem path, or free-form expression is accepted.
- Schema validation and semantic validation are separate and both run inside `run_spec()`, even if the caller validated earlier.
- MCP-visible components come from an explicit allowlist, not from every Python-registered component.
- A spec-level risk constraint may tighten but never weaken the engine-level `RiskPolicy`.
- Component compilation must be deterministic; canonical spec JSON produces a stable SHA-256 fingerprint.
- Python strategies and compiled strategies use the same `FeatureBuilder`, `Strategy`, `OrderIntent`, risk, broker, ledger, and result types.
- Experiment storage never requires or stores hidden model reasoning.
- Every shell command is prefixed with `rtk`.

## File Map

```text
src/pybacktest/
├── specs/
│   ├── __init__.py              # Public spec API
│   ├── models.py                # Strict Pydantic unions
│   ├── issues.py                # Machine-readable validation issues
│   ├── registry.py              # Component registry/catalog/allowlist
│   ├── validation.py            # Semantic validation and graph checks
│   ├── compiler.py              # StrategySpec -> CompiledStrategyBundle
│   └── runtime.py               # Compiled rule/action strategy
├── risk/
│   └── composite.py             # Engine/spec risk intersection
└── application/
    ├── requests.py              # SpecBacktestRequest
    ├── service.py               # run_spec extension
    └── experiments.py           # Experiment DTOs/service
src/pybacktest/adapters/experiments/
├── __init__.py
└── sqlite.py                    # Local lineage store
tests/specs/
tests/experiments/
```

## Task 1: Strict Versioned StrategySpec Schema

**Files:**
- Create: `src/pybacktest/specs/__init__.py`
- Create: `src/pybacktest/specs/models.py`
- Create: `src/pybacktest/specs/issues.py`
- Create: `tests/specs/test_schema.py`
- Create: `tests/specs/fixtures/ma_cross_v1.json`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: canonical `InstrumentId` string syntax from core.
- Produces:
  - `StrategySpec`
  - feature unions `SourceFeatureSpec`, `SmaFeatureSpec`,
    `EmaFeatureSpec`, `LagFeatureSpec`
  - condition unions `CrossesAboveSpec`, `CrossesBelowSpec`,
    `GreaterThanSpec`, `LessThanSpec`, `AllSpec`, `AnySpec`, `NotSpec`
  - action unions `TargetWeightActionSpec`, `TargetQuantityActionSpec`,
    `MarketOrderActionSpec`, `LimitOrderActionSpec`
  - sizer union `DefaultOrderSizerSpec`
  - risk unions `MaxPositionWeightSpec`, `MaxLeverageSpec`,
    `AllowShortSpec`
  - `SpecValidationIssue(code, path, message)`

- [ ] **Step 1: Add the optional spec dependency**

Add:

```toml
[project.optional-dependencies]
spec = ["pydantic>=2.12,<3"]
```

Ensure the dev dependency group also contains `pydantic>=2.12,<3`, then run:

```bash
rtk proxy uv lock
```

- [ ] **Step 2: Write failing schema tests**

```python
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pybacktest.specs import StrategySpec


FIXTURE = Path(__file__).parent / "fixtures" / "ma_cross_v1.json"


def test_ma_cross_fixture_parses_as_version_one():
    spec = StrategySpec.model_validate_json(FIXTURE.read_text())
    assert spec.spec_version == "1"
    assert spec.universe == ("XNAS:AAPL", "XNAS:MSFT")
    assert spec.features["fast"].type == "sma"
    assert spec.rules[0].then.type == "target_weight"
    assert spec.sizer.type == "default"


def test_unknown_fields_and_import_paths_are_rejected():
    payload = json.loads(FIXTURE.read_text())
    payload["features"]["fast"]["callable"] = "evil.module:run"
    with pytest.raises(ValidationError, match="callable"):
        StrategySpec.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("features", "fast", "type"), "percentage"),
        (("rules", 0, "then", "type"), "python"),
    ],
)
def test_unknown_feature_and_action_discriminators_are_rejected(path, value):
    payload = json.loads(FIXTURE.read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(payload)


@pytest.mark.parametrize("version", ["0", "2", "", 1])
def test_only_string_version_one_is_accepted(version):
    payload = json.loads(FIXTURE.read_text())
    payload["spec_version"] = version
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(payload)
```

Create the fixture with the exact approved moving-average strategy:

```json
{
  "spec_version": "1",
  "name": "ma-cross",
  "universe": ["XNAS:AAPL", "XNAS:MSFT"],
  "features": {
    "fast": {
      "type": "sma",
      "instrument": "XNAS:AAPL",
      "source": "close",
      "window": 20
    },
    "slow": {
      "type": "sma",
      "instrument": "XNAS:AAPL",
      "source": "close",
      "window": 60
    }
  },
  "rules": [
    {
      "when": {
        "type": "crosses_above",
        "left": {"feature": "fast"},
        "right": {"feature": "slow"}
      },
      "then": {
        "type": "target_weight",
        "instrument": "XNAS:AAPL",
        "weight": 0.5
      }
    }
  ],
  "sizer": {"type": "default"},
  "risk": [
    {"type": "max_position_weight", "value": 0.6},
    {"type": "allow_short", "value": false}
  ]
}
```

- [ ] **Step 3: Verify tests fail**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_schema.py -q
```

Expected: collection FAIL because `pybacktest.specs` does not exist.

- [ ] **Step 4: Implement strict discriminated models**

Use a common base:

```python
class StrictSpecModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )
```

Use `Literal` discriminator values and `Annotated[..., Field(discriminator="type")]`.
Use tuples for ordered immutable collections. Constrain:

- feature/rule names: regex `^[a-z][a-z0-9_]{0,63}$`
- windows/spans/lag: `1..100_000`
- weights: `-10..10`
- quantities: finite, non-zero Decimal
- limit price: finite, positive Decimal
- risk leverage/weight: finite and positive
- universe: `1..1_000` entries
- rules: `1..10_000` entries
- exactly one required sizer object
- risk: `0..16` entries with no duplicate risk type

Use `mode="before"` validators on numeric fields to reject booleans and
numeric strings before Decimal/int conversion. JSON numbers are accepted;
`"20"` and `true` are not valid substitutes for a window or weight. Add both
cases to `test_schema.py`.

Operands are exactly:

```python
class FeatureOperand(StrictSpecModel):
    feature: str


class ConstantOperand(StrictSpecModel):
    value: float
```

No expression string type exists.

- [ ] **Step 5: Expose canonical JSON and fingerprint helpers**

Add:

```python
def canonical_spec_json(spec: StrategySpec) -> str:
    return json.dumps(
        spec.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
            round_trip=True,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def strategy_spec_fingerprint(spec: StrategySpec) -> str:
    payload = json.dumps(
        spec.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
```

Test that dictionary key order does not change the fingerprint.

- [ ] **Step 6: Verify and commit**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_schema.py -q
rtk proxy uv run ruff check src/pybacktest/specs tests/specs
rtk proxy uv run ty check src/pybacktest/specs
rtk git add pyproject.toml uv.lock src/pybacktest/specs tests/specs
rtk git commit -m "feat: add strict strategy spec schema"
```

## Task 2: Component Registry, Catalog, and Allowlist

**Files:**
- Create: `src/pybacktest/specs/registry.py`
- Create: `tests/specs/test_registry.py`

**Interfaces:**
- Consumes: spec model classes and core compiler target interfaces.
- Produces:
  - `ComponentKind`
  - `ComponentDescriptor(name, kind, schema, description, output_unit, lookback)`
  - `ComponentRegistration(descriptor, compiler_factory)`
  - `ComponentRegistry.register(registration)`
  - `ComponentRegistry.resolve(kind, name)`
  - `ComponentRegistry.all_keys() -> frozenset[tuple[ComponentKind, str]]`
  - `ComponentRegistry.catalog(allowlist) -> tuple[ComponentDescriptor, ...]`
  - `default_component_registry()`

- [ ] **Step 1: Write failing registry-security tests**

```python
import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.specs.registry import (
    ComponentDescriptor,
    ComponentKind,
    ComponentRegistry,
    ComponentRegistration,
    default_component_registry,
)


def test_duplicate_component_key_is_rejected():
    registry = ComponentRegistry()
    registration = ComponentRegistration(
        descriptor=ComponentDescriptor(
            name="sma",
            kind=ComponentKind.FEATURE,
            schema={"type": "object"},
            description="Simple moving average.",
            output_unit="same_as_input",
            lookback="window",
        ),
        compiler_factory=lambda spec, context: ("sma", spec, context),
    )
    registry.register(registration)
    with pytest.raises(ConfigurationError, match="duplicate"):
        registry.register(registration)


def test_catalog_exposes_only_explicit_allowlist():
    registry = default_component_registry()
    catalog = registry.catalog(
        allowlist={
            (ComponentKind.FEATURE, "sma"),
            (ComponentKind.ACTION, "target_weight"),
        }
    )
    assert {(item.kind, item.name) for item in catalog} == {
        (ComponentKind.FEATURE, "sma"),
        (ComponentKind.ACTION, "target_weight"),
    }


def test_registry_has_no_import_path_registration_api():
    assert not hasattr(ComponentRegistry, "register_import_path")
    assert not hasattr(ComponentRegistry, "load_module")
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_registry.py -q
```

- [ ] **Step 3: Implement a frozen-after-build registry**

Key registrations by `(ComponentKind, name)`. Reject duplicate names, names
that fail the component regex, empty descriptions, missing schemas, and
registration after `freeze()`.

`catalog(allowlist)` must:

1. reject allowlist keys not present in the registry;
2. return only descriptors, never compiler callables;
3. sort by kind then name;
4. return immutable tuples and copied schemas.

`default_component_registry()` registers the exact V1 set:

```text
feature: source, sma, ema, lag
condition: crosses_above, crosses_below, greater_than, less_than, all, any, not
action: target_weight, target_quantity, market_order, limit_order
sizer: default
risk: max_position_weight, max_leverage, allow_short
```

Use direct Python factory references defined in code. Do not discover entry
points in this default registry. A later Python-only extension may build a
separate registry explicitly.

- [ ] **Step 4: Verify and commit**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_registry.py -q
rtk proxy uv run ruff check src/pybacktest/specs/registry.py tests/specs/test_registry.py
rtk proxy uv run ty check src/pybacktest/specs/registry.py
rtk git add src/pybacktest/specs/registry.py tests/specs/test_registry.py
rtk git commit -m "feat: add allowlisted strategy component registry"
```

## Task 3: Semantic Validation and Complete Issue Reporting

**Files:**
- Create: `src/pybacktest/specs/validation.py`
- Create: `tests/specs/test_validation.py`

**Interfaces:**
- Consumes: `StrategySpec`, `ComponentRegistry`, allowlist,
  `Mapping[InstrumentId, Instrument]`.
- Produces:
  - `SpecValidationIssue(code, path, message)`
  - `StrategySpecValidator.validate(spec, instruments, allowlist) -> tuple[SpecValidationIssue, ...]`
  - `StrategySpecValidator.require_valid(...) -> None`
  - `StrategySpecError(issues)`

- [ ] **Step 1: Write failing multi-issue validation tests**

```python
import json
from pathlib import Path

import pytest

from pybacktest.specs import StrategySpec
from pybacktest.specs.issues import StrategySpecError
from pybacktest.specs.registry import ComponentKind, default_component_registry
from pybacktest.specs.validation import StrategySpecValidator
from tests.factories import aapl


FIXTURE = Path(__file__).parent / "fixtures" / "ma_cross_v1.json"


def test_validator_returns_all_reference_and_allowlist_issues():
    payload = json.loads(FIXTURE.read_text())
    payload["features"]["fast"]["instrument"] = "XNAS:UNKNOWN"
    payload["rules"][0]["when"]["left"]["feature"] = "missing"
    spec = StrategySpec.model_validate(payload)
    validator = StrategySpecValidator(default_component_registry())
    issues = validator.validate(
        spec,
        instruments={aapl().id: aapl()},
        allowlist={(ComponentKind.FEATURE, "sma")},
    )
    assert {issue.code for issue in issues} == {
        "unknown_instrument",
        "unknown_feature_reference",
        "component_not_allowed",
    }
    assert all(issue.path for issue in issues)


def test_require_valid_raises_one_error_with_structured_issues():
    spec = StrategySpec.model_validate_json(FIXTURE.read_text())
    validator = StrategySpecValidator(default_component_registry())
    with pytest.raises(StrategySpecError) as caught:
        validator.require_valid(spec, instruments={}, allowlist=set())
    assert caught.value.issues
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_validation.py -q
```

- [ ] **Step 3: Implement deterministic semantic passes**

Run passes in this exact order and accumulate issues:

1. canonical universe parse, duplicate IDs, and metadata presence;
2. component existence and allowlist;
3. feature instrument membership and source field units;
4. feature dependency existence and cycle detection;
5. condition operand/reference existence and unit compatibility;
6. action instrument membership, weight/quantity/price units;
7. sizer component existence, allowlist membership, and parameter units;
8. risk consistency (`allow_short=false` plus negative target is an issue);
9. each action against declared spec risk bounds; do not sum targets across
   rules because mutually exclusive/runtime conditions cannot be inferred
   statically.

Issue paths use JSON-pointer-like tuples such as
`("rules", 0, "when", "left", "feature")`. Sort issues by path then code so
tool responses and tests are stable.

`StrategySpecError` stores the tuple and formats a concise multi-line display;
callers consume `.issues`, never parse its string.

- [ ] **Step 4: Add graph-cycle and unit tests**

Construct a spec in which feature A depends on B and B on A; assert
`feature_cycle`. Compare a price feature to a volume feature and assert
`incompatible_units`. Assert a constant operand can compare to either after
the component declares the constant's expected unit.

- [ ] **Step 5: Verify and commit**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_validation.py -q
rtk proxy uv run ruff check src/pybacktest/specs tests/specs
rtk proxy uv run ty check src/pybacktest/specs
rtk git add src/pybacktest/specs/issues.py src/pybacktest/specs/validation.py tests/specs/test_validation.py
rtk git commit -m "feat: validate strategy spec semantics"
```

## Task 4: Compile Features, Conditions, and Actions Without `eval`

**Files:**
- Create: `src/pybacktest/specs/runtime.py`
- Create: `src/pybacktest/specs/compiler.py`
- Create: `tests/specs/test_compiler.py`
- Create: `tests/specs/test_compiler_security.py`

**Interfaces:**
- Consumes: validated `StrategySpec`, frozen registry, core `FeatureBuilder`,
  `Strategy`, `StrategyContext`, and intent types.
- Produces:
  - `CompiledStrategyBundle(strategy, requested_sizer, requested_risk, spec_fingerprint)`
  - `StrategySpecCompiler.compile(spec, instruments, allowlist) -> CompiledStrategyBundle`
  - private immutable condition/action runtime nodes

- [ ] **Step 1: Write the failing compiler behavior test**

```python
from pathlib import Path

from pybacktest.specs import StrategySpec
from pybacktest.specs.compiler import StrategySpecCompiler
from pybacktest.specs.registry import default_component_registry
from decimal import Decimal

from tests.factories import aapl, feature_context, market_slice


FIXTURE = Path(__file__).parent / "fixtures" / "ma_cross_v1.json"


def test_ma_cross_compiles_to_target_weight_only_on_cross():
    spec = StrategySpec.model_validate_json(FIXTURE.read_text())
    bundle = StrategySpecCompiler(default_component_registry()).compile(
        spec,
        instruments={aapl().id: aapl()},
        allowlist=default_component_registry().all_keys(),
    )
    before_context = feature_context(
        previous={"fast": 99, "slow": 100},
        current={"fast": 100, "slow": 101},
    )
    crossed_context = feature_context(
        previous={"fast": 99, "slow": 100},
        current={"fast": 101, "slow": 100},
    )
    before = bundle.strategy.on_bar(
        before_context,
        market_slice(before_context.timestamp),
    )
    crossed = bundle.strategy.on_bar(
        crossed_context,
        market_slice(crossed_context.timestamp),
    )
    assert before == ()
    assert len(crossed) == 1
    assert crossed[0].instrument == aapl().id
    assert crossed[0].weight == Decimal("0.5")
    assert crossed[0].reason.code == "strategy_spec_rule"
    assert crossed[0].reason.details["rule_index"] == 0
```

`feature_context()` returns a `StrategyContext`; `market_slice(timestamp)`
returns the separate `MarketSlice` required by the public strategy protocol.
The test does not call private condition/action nodes.

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_compiler.py -q
```

- [ ] **Step 3: Implement compile-time feature mapping**

Map feature specs directly to `FeatureBuilder` methods:

```python
if isinstance(spec, SourceFeatureSpec):
    node = builder.source(name, InstrumentId.parse(spec.instrument), spec.source)
elif isinstance(spec, SmaFeatureSpec):
    source = builder.source(
        f"__{name}_source",
        InstrumentId.parse(spec.instrument),
        spec.source,
    )
    node = builder.sma(name, source, spec.window)
elif isinstance(spec, EmaFeatureSpec):
    source = builder.source(
        f"__{name}_source",
        InstrumentId.parse(spec.instrument),
        spec.source,
    )
    node = builder.ema(name, source, spec.span)
elif isinstance(spec, LagFeatureSpec):
    node = builder.lag(name, compiled[spec.input_feature], spec.periods)
else:
    raise AssertionError("Validated feature union was not exhaustive.")
```

Use registry factory dispatch in production rather than a public dynamic
import. Keep the exhaustive type check and test every registered component.
Compile `DefaultOrderSizerSpec(type="default")` through the allowlisted sizer
registration to a fresh `DefaultOrderSizer` stored on
`CompiledStrategyBundle.requested_sizer`; no implicit sizer is inserted by the
compiler.

- [ ] **Step 4: Implement immutable condition/action runtime nodes**

Conditions receive only the current/previous `FeatureView`. Crosses compare
previous and current values and return false if any input is NaN. Boolean
conditions short-circuit in declared order.

Actions construct core intents and include an immutable reason mapping:

```python
{
    "spec_fingerprint": fingerprint,
    "rule_index": rule_index,
    "condition_type": condition.type,
    "feature_values": observed_values,
}
```

Never call `eval`, `exec`, `compile`, `importlib`, `__import__`, or deserialize
pickle. Add a source scan test:

```python
def test_compiler_source_contains_no_dynamic_execution():
    source = Path(compiler_module.__file__).read_text()
    for forbidden in ("eval(", "exec(", "__import__(", "importlib", "pickle"):
        assert forbidden not in source
```

- [ ] **Step 5: Verify compiler behavior and security**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_compiler.py tests/specs/test_compiler_security.py -q
rtk proxy uv run ruff check src/pybacktest/specs tests/specs
rtk proxy uv run ty check src/pybacktest/specs
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/specs/runtime.py src/pybacktest/specs/compiler.py tests/specs/test_compiler.py tests/specs/test_compiler_security.py
rtk git commit -m "feat: compile safe strategy specs"
```

## Task 5: Risk Intersection and Python/Spec Parity

**Files:**
- Create: `src/pybacktest/risk/composite.py`
- Create: `tests/specs/test_risk_composition.py`
- Create: `tests/specs/test_python_parity.py`

**Interfaces:**
- Consumes: `CompiledStrategyBundle.requested_risk`, engine `RiskPolicy`,
  core reference `MovingAverageCross`.
- Produces:
  - `CompositeRiskPolicy(engine_policy, spec_policy)`
  - stable combined `RiskDecision`
  - parity guarantee between reference Python and equivalent spec strategy

- [ ] **Step 1: Write failing risk-intersection tests**

```python
from decimal import Decimal

from pybacktest.risk import LongShortRisk
from pybacktest.risk.composite import CompositeRiskPolicy
from tests.factories import proposed_buy_order, proposed_short_order, risk_context


def test_spec_risk_can_tighten_but_not_weaken_engine_risk():
    engine = LongShortRisk(
        max_leverage=Decimal("1.0"),
        max_position_weight=Decimal("0.40"),
        allow_short=False,
    )
    requested = LongShortRisk(
        max_leverage=Decimal("2.0"),
        max_position_weight=Decimal("0.25"),
        allow_short=True,
    )
    policy = CompositeRiskPolicy(engine, requested)
    decision = policy.evaluate(
        proposed_buy_order(weight="0.50"),
        risk_context(),
    )
    assert policy.effective_max_position_weight == Decimal("0.25")
    assert policy.effective_max_leverage == Decimal("1.0")
    assert policy.effective_allow_short is False
    assert decision.final_quantity.value == Decimal("25")
    assert decision.codes == ("spec.max_position_weight",)

    short_decision = policy.evaluate(proposed_short_order(), risk_context())
    assert short_decision.codes == ("engine.short_not_allowed",)
```

- [ ] **Step 2: Implement deterministic risk intersection**

For maximum limits choose `min(engine, spec)`; for boolean permissions use
logical AND. Evaluate the effective policy once, but retain provenance for
every contributing constraint in sorted code order. A missing spec risk means
the engine policy alone.

- [ ] **Step 3: Add full-run Python/spec parity test**

Run the core `MovingAverageCross(fast=20, slow=60)` and a dedicated
bidirectional spec containing both crosses-above→long and
crosses-below→flat rules against the same frozen dataset, broker, risk,
initial cash, and seed. Assert identical execution projections:

```python
assert execution_projection(spec_result.orders) == execution_projection(
    python_result.orders
)
assert execution_projection(spec_result.fills) == execution_projection(
    python_result.fills
)
assert spec_result.snapshots == python_result.snapshots
assert spec_result.summary == python_result.summary
```

`execution_projection()` retains timestamp, instrument, side, type, quantity,
limit, time-in-force, status, and active-from, while normalizing run-scoped IDs
and excluding only strategy-specific `DecisionReason` provenance. Separately
assert both reason payloads contain the observed fast/slow values and rule
outcome. Only strategy identity, spec fingerprint, rule index, and run-scoped
IDs may differ.

- [ ] **Step 4: Verify and commit**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs/test_risk_composition.py tests/specs/test_python_parity.py -q
rtk proxy uv run ruff check src/pybacktest/risk/composite.py tests/specs
rtk proxy uv run ty check src/pybacktest/risk/composite.py
rtk git add src/pybacktest/risk/composite.py tests/specs/test_risk_composition.py tests/specs/test_python_parity.py
rtk git commit -m "feat: compose spec and engine risk"
```

## Task 6: `BacktestService.run_spec()` and Spec Artifacts

**Files:**
- Modify: `src/pybacktest/application/requests.py`
- Modify: `src/pybacktest/application/service.py`
- Modify: `src/pybacktest/results/models.py`
- Modify: `src/pybacktest/results/serialization.py`
- Create: `tests/application/test_run_spec.py`

**Interfaces:**
- Consumes: validator/compiler/risk composition and core engine.
- Produces:
  - `SpecBacktestRequest(spec, simulation, run_id=None)`
  - `BacktestService.run_spec(request) -> BacktestResult`
  - `RunManifest.strategy_spec_fingerprint`
  - `strategy_spec.json` artifact

- [ ] **Step 1: Write a failing revalidation test**

```python
import pytest

from pybacktest.application.requests import SpecBacktestRequest
from pybacktest.specs.issues import StrategySpecError
from tests.factories import invalid_reference_spec, simulation_request, spec_service


def test_run_spec_revalidates_even_when_caller_skips_validate():
    service = spec_service()
    request = SpecBacktestRequest(
        spec=invalid_reference_spec(),
        simulation=simulation_request(),
    )
    with pytest.raises(StrategySpecError) as caught:
        service.run_spec(request)
    assert caught.value.issues[0].code == "unknown_feature_reference"
    assert service.engine_run_count == 0
```

- [ ] **Step 2: Verify test fails**

```bash
rtk proxy uv run --extra spec python -m pytest tests/application/test_run_spec.py -q
```

- [ ] **Step 3: Implement the application use case**

`run_spec()` must:

1. schema-validate input if passed as a mapping;
2. semantic-validate against the exact dataset instrument catalog and
   allowlist;
3. compile;
4. intersect spec and engine risk;
5. call the same
   `BacktestEngine.run(..., order_sizer=bundle.requested_sizer, risk_policy=composite_policy)`
   used by Python strategies;
6. attach spec/compiler/schema fingerprints to `RunManifest`;
7. serialize canonical `strategy_spec.json` with artifacts.

Inject the instrument catalog, validator, compiler, frozen registry, MCP
allowlist, and engine-default risk policy into `BacktestService` at
construction. `run_spec()` accepts only the request, so callers cannot swap
the trust policy per call.

No validation result cache may bypass a run-time recheck; caching parsed schema
objects by fingerprint is allowed only after checking the same registry,
allowlist, and instrument-catalog fingerprints.

- [ ] **Step 4: Verify and commit**

```bash
rtk proxy uv run --extra spec python -m pytest tests/application/test_run_spec.py tests/specs -q
rtk proxy uv run ruff check src tests/application tests/specs
rtk proxy uv run ty check src/pybacktest
rtk git add src/pybacktest/application src/pybacktest/results tests/application
rtk git commit -m "feat: run validated strategy specs"
```

## Task 7: Reproducible AI Experiment Lineage

**Files:**
- Create: `src/pybacktest/application/experiments.py`
- Create: `src/pybacktest/adapters/experiments/__init__.py`
- Create: `src/pybacktest/adapters/experiments/sqlite.py`
- Create: `tests/experiments/test_store.py`
- Create: `tests/experiments/test_lineage.py`

**Interfaces:**
- Consumes: `StrategySpec`, validation issues, `RunId`, summary metrics,
  dataset/spec fingerprints.
- Produces:
  - `ExperimentId`, `IterationId`
  - `ExperimentRecord`
  - `IterationRecord`
  - `ExperimentStore` protocol
  - `SQLiteExperimentStore`
  - `ExperimentService.record_validation()`
  - `ExperimentService.record_run()`
  - `ExperimentService.lineage(experiment_id)`
  - `ExperimentService.training_trajectory(experiment_id)`
  - immutable `TrainingTrajectory` and `TrainingStep`

- [ ] **Step 1: Write failing persistence and lineage tests**

```python
from pybacktest.application.experiments import ExperimentService
from pybacktest.adapters.experiments.sqlite import SQLiteExperimentStore
from tests.factories import ma_cross_spec, run_summary


RUN_1 = f"run_{'1' * 32}"
RUN_2 = f"run_{'2' * 32}"


def test_iteration_lineage_round_trips_without_hidden_reasoning(tmp_path):
    store = SQLiteExperimentStore(tmp_path / "experiments.sqlite3")
    service = ExperimentService(store)
    experiment = service.create("agent-search")
    first = service.record_run(
        experiment_id=experiment.id,
        agent_session_id="session-1",
        parent_run_id=None,
        spec=ma_cross_spec(fast=20, slow=60),
        dataset_fingerprint="dataset-a",
        run_summary=run_summary(run_id=RUN_1, sharpe=1.0),
    )
    second = service.record_run(
        experiment_id=experiment.id,
        agent_session_id="session-1",
        parent_run_id=first.run_id,
        spec=ma_cross_spec(fast=10, slow=50),
        dataset_fingerprint="dataset-a",
        run_summary=run_summary(run_id=RUN_2, sharpe=1.2),
    )

    lineage = service.lineage(experiment.id)
    assert [item.run_id for item in lineage] == [RUN_1, RUN_2]
    assert lineage[1].parent_run_id == RUN_1
    assert "reasoning" not in lineage[1].model_dump()
    assert "prompt" not in lineage[1].model_dump()

    trajectory = service.training_trajectory(experiment.id)
    assert [step.run_id for step in trajectory.steps] == [RUN_1, RUN_2]
    assert trajectory.steps[1].parent_run_id == RUN_1
    assert trajectory.steps[1].dataset_fingerprint == "dataset-a"
    assert "reasoning" not in trajectory.model_dump_json()
    assert "prompt" not in trajectory.model_dump_json()
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra spec python -m pytest tests/experiments -q
```

- [ ] **Step 3: Implement explicit experiment DTOs and store protocol**

An iteration contains only:

```text
iteration_id
experiment_id
agent_session_id
parent_run_id
spec_fingerprint
canonical_strategy_spec
validation_issues
dataset_fingerprint
run_id
summary_metrics
rejection_count
artifact_id
feedback
created_at
```

Natural-language feedback is a separate optional `feedback` string accepted
only when the caller supplies it. There is no reasoning/chain-of-thought field.

`TrainingStep` is a lossless, JSON-serializable projection containing the
submitted canonical spec, structured validation issues, dataset/spec
fingerprints, parent/run IDs, summary metric definitions and values,
rejection count, artifact ID, and optional explicit feedback.
`TrainingTrajectory` preserves iteration order and experiment identity. It
does not invent rewards, labels, prompts, or model reasoning; a later training
project chooses those transformations from this stable projection.

- [ ] **Step 4: Implement SQLite schema and transactions**

Create normalized `experiments` and `iterations` tables with foreign keys,
unique IDs, parent-run index, and canonical JSON text. Enable
`PRAGMA foreign_keys=ON`, WAL mode, a documented busy timeout, and
transactions. The constructor stores the path only; the first operation
creates the database. Open one connection per operation so MCP worker threads
never share a thread-bound SQLite connection.

Reject:

- unknown experiment;
- parent run outside the experiment;
- duplicate run ID;
- a parent that would create a lineage cycle;
- non-canonical spec JSON.

Return immutable DTOs ordered by creation sequence then iteration ID.
Add a concurrent-record test proving two child iterations either both commit
with distinct IDs or return a typed uniqueness error, never a partial row or
`sqlite3.ProgrammingError`.

- [ ] **Step 5: Verify complete plan-2 behavior**

```bash
rtk proxy uv run --extra spec python -m pytest tests/specs tests/application tests/experiments -q
rtk proxy uv run ruff check src tests
rtk proxy uv run ruff format --check src tests
rtk proxy uv run ty check src/pybacktest
rtk git diff --check
```

- [ ] **Step 6: Document and commit**

Add a README section showing:

```text
StrategySpec → validate → run_spec → compare metrics → create child iteration
```

Explain allowlists, spec fingerprinting, risk intersection, stored fields,
`training_trajectory()`, and the explicit absence of arbitrary code and hidden
reasoning.

```bash
rtk git add src/pybacktest/application/experiments.py src/pybacktest/adapters/experiments tests/experiments README.md
rtk git commit -m "feat: record strategy experiment lineage"
```

## StrategySpec Plan Completion Check

Run:

```bash
rtk git status --short
rtk proxy uv run --extra spec python -m pytest -q
rtk proxy uv run ruff check src tests
rtk proxy uv run ty check src/pybacktest
```

The worktree must be clean. Plan 3 consumes:

```python
StrategySpec
SpecValidationIssue
StrategySpecValidator
ComponentDescriptor
ComponentRegistry.catalog
BacktestService.run_spec
ExperimentService
BacktestResult
LocalArtifactStore
```
