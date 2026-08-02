# Pybacktest V2 MCP Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the approved safe `StrategySpec` workflow through a local stdio MCP server with six bounded tools and six read-only resources (one concrete resource and five templates), while keeping MCP optional and preventing arbitrary code, path, URL, dataset, or component access.

**Architecture:** A transport-independent `MCPApplication` owns validation, job submission, result lookup, comparison, and trace explanation. It depends only on an allowlisted `DatasetCatalog`, an exact dataset-to-`BacktestService` map, the StrategySpec component catalog, a bounded `RunJobRegistry`, and optional experiment lineage services. A thin adapter registers typed Pydantic inputs and outputs on the official MCP Python SDK v2 `MCPServer`; production uses stdio, while tests use the SDK's in-memory `Client`.

**Tech Stack:** Python 3.11+, Pybacktest V2 core and StrategySpec interfaces, Pydantic 2, official MCP Python SDK 2.x, stdlib concurrent.futures/threading, pytest, AnyIO pytest support, Ruff, ty, uv.

## Global Constraints

- Complete and verify `2026-07-29-pybacktest-v2-core.md` and `2026-07-29-pybacktest-v2-strategy-spec.md` first.
- Use the official MCP Python SDK v2 API: `MCPServer` from `mcp.server` and the in-memory `Client` from `mcp`.
- `pybacktest` core imports must continue to work when the MCP extra is not installed.
- Version `0.2.0` exposes local stdio only; do not add HTTP transport, remote authentication, or multi-user claims.
- MCP accepts stable `dataset_id` values from server configuration. It never accepts a filesystem path, URL, Python source, import path, callable, SQL, shell command, or environment-variable name.
- `run_backtest` always repeats schema and semantic validation immediately before execution.
- Component and dataset catalogs are explicit allowlists. Python-side registry membership alone does not make a component MCP-visible.
- Reject work above configured instrument, bar-observation, concurrent-run, input-byte, artifact-byte, comparison-count, and inline-row limits.
- Tool outputs use frozen strict Pydantic models and stable machine-readable error codes.
- Never return credentials, environment values, hostnames, stack traces, internal filesystem paths, or hidden model reasoning.
- Large tables remain in artifacts; resources return bounded previews plus row counts and a `truncated` flag.
- MCP tools do not mutate an already-completed run. Each accepted execution receives a new `RunId`.
- Every shell command is prefixed with `rtk`.

## File Map

```text
src/pybacktest/
├── application/
│   └── run_repository.py       # Run state/result lookup protocol
├── adapters/
│   └── runs/
│       ├── __init__.py
│       └── memory.py           # Thread-safe bounded local run registry
└── mcp/
    ├── __init__.py             # Optional adapter exports only
    ├── __main__.py             # python -m pybacktest.mcp
    ├── config.py               # Quotas and allowlisted dataset bindings
    ├── models.py               # Strict MCP input/output DTOs
    ├── errors.py               # Stable public MCP errors
    ├── application.py          # Transport-independent use cases
    ├── artifact_reader.py      # Safe persisted result projections
    ├── comparison.py           # Metric/config/lineage comparison
    ├── resources.py            # Bounded resource projections
    └── server.py               # MCPServer registration and stdio entrypoint
tests/mcp/
├── conftest.py
├── test_optional_boundary.py
├── test_catalog_and_validation.py
├── test_run_lifecycle.py
├── test_compare_and_explain.py
├── test_resources.py
├── test_security.py
└── test_stdio_entrypoint.py
```

## Task 1: Optional MCP Boundary, Configuration, and Dataset Allowlist

**Files:**
- Create: `src/pybacktest/mcp/__init__.py`
- Create: `src/pybacktest/mcp/config.py`
- Create: `src/pybacktest/mcp/errors.py`
- Create: `tests/mcp/test_optional_boundary.py`
- Create: `tests/mcp/test_security.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `InstrumentId`, `Timeframe`, `DateRange`,
  `MarketDataSource`, and dataset fingerprints from the core plan.
- Produces:
  - `MCPSettings`
  - `DatasetDescriptor`
  - `DatasetBinding`
  - `DatasetCatalog`
  - `MCPPublicError(code, message, details)`

- [ ] **Step 1: Add the optional MCP dependency and entrypoint**

Add:

```toml
[project.optional-dependencies]
mcp = [
  "mcp>=2,<3",
  "pydantic>=2.12,<3",
]

[project.scripts]
pybacktest-mcp = "pybacktest.mcp.server:main"
```

Keep the existing `spec` extra. Resolve the lockfile:

```bash
rtk proxy uv lock
```

- [ ] **Step 2: Write failing optional-boundary and configuration tests**

```python
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from pybacktest.mcp.config import MCPSettings


def test_importing_core_does_not_import_mcp_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import pybacktest; "
                "print(json.dumps('mcp' in sys.modules))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) is False


def test_settings_require_positive_bounded_limits():
    with pytest.raises(ValidationError):
        MCPSettings(
            max_instruments=0,
            max_bar_observations=1_000,
            max_concurrent_runs=1,
            max_request_bytes=100_000,
            max_response_bytes=100_000,
            max_artifact_bytes=1_000_000,
            max_inline_rows=100,
            max_compare_runs=5,
            max_trace_entries=500,
        )
```

- [ ] **Step 3: Verify the tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_optional_boundary.py -q
```

Expected: collection FAIL because `pybacktest.mcp.config` does not exist.

- [ ] **Step 4: Implement frozen configuration and catalog types**

Use strict positive integers and conservative defaults:

```python
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import DateRange, Timeframe
from pybacktest.ports.data import MarketDataSource


class MCPSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_instruments: int = Field(default=100, ge=1, le=10_000)
    max_bar_observations: int = Field(
        default=1_000_000,
        ge=1,
        le=100_000_000,
    )
    max_concurrent_runs: int = Field(default=2, ge=1, le=32)
    max_request_bytes: int = Field(default=1_000_000, ge=1_024)
    max_response_bytes: int = Field(default=1_000_000, ge=1_024)
    max_artifact_bytes: int = Field(default=250_000_000, ge=1_024)
    max_inline_rows: int = Field(default=200, ge=1, le=2_000)
    max_compare_runs: int = Field(default=10, ge=2, le=50)
    max_trace_entries: int = Field(default=500, ge=1, le=5_000)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    dataset_id: str
    display_name: str
    instruments: tuple[InstrumentId, ...]
    timeframes: tuple[Timeframe, ...]
    available_period: DateRange
    fingerprint: str


class ObservationEstimator(Protocol):
    def estimate(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> int:
        """Return a metadata-only conservative observation count."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DatasetBinding:
    descriptor: DatasetDescriptor
    source: MarketDataSource
    instrument_catalog: Mapping[InstrumentId, Instrument]
    observation_estimator: ObservationEstimator


class DatasetCatalog:
    def __init__(self, bindings: tuple[DatasetBinding, ...]) -> None: ...
    def list_descriptors(self) -> tuple[DatasetDescriptor, ...]: ...
    def resolve(self, dataset_id: str) -> DatasetBinding: ...
```

The constructor rejects empty/duplicate IDs and invalid IDs. Use the public ID
regex `^[a-z0-9][a-z0-9._-]{0,63}$`. `resolve()` performs exact-key lookup and
raises `MCPPublicError(code="dataset_not_allowed", ...)`; it never interprets
the ID as a path. Copy mappings into `MappingProxyType` and ordered values into
tuples so operator configuration cannot mutate the live allowlist.

- [ ] **Step 5: Add path/URL confusion tests**

Parameterize `../prices`, `/tmp/prices`, `file:///tmp/prices`,
`https://example.test/prices`, `${HOME}`, and `XNAS:AAPL` as dataset IDs.
Assert every value is rejected rather than normalized or opened. Add a fake
source that increments a counter and assert rejected lookups never call it.

- [ ] **Step 6: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_optional_boundary.py tests/mcp/test_security.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/mcp
rtk git add pyproject.toml uv.lock src/pybacktest/mcp tests/mcp
rtk git commit -m "feat: add bounded MCP configuration"
```

## Task 2: Strict MCP DTOs and Transport-Independent Catalog/Validation

**Files:**
- Create: `src/pybacktest/mcp/models.py`
- Create: `src/pybacktest/mcp/application.py`
- Create: `tests/mcp/conftest.py`
- Create: `tests/mcp/test_catalog_and_validation.py`

**Interfaces:**
- Consumes: `StrategySpec`, `StrategySpecValidator`,
  `ComponentRegistry.catalog()`, MCP component allowlist, and
  `DatasetCatalog`.
- Produces:
  - `ComponentCatalogOutput`
  - `ValidateStrategySpecInput`, `ValidateStrategySpecOutput`
  - generic `ToolEnvelope[T]`
  - `MCPBacktestConfig`
  - `MCPApplication.list_strategy_components()`
  - `MCPApplication.validate_strategy_spec()`
  - `ConfiguredBacktestServices.resolve(dataset_id) -> BacktestService`

- [ ] **Step 1: Write failing catalog and validation tests**

```python
from pybacktest.mcp.models import ValidateStrategySpecInput
from tests.factories import ma_cross_spec


def test_catalog_contains_only_explicitly_mcp_allowed_components(mcp_app):
    output = mcp_app.list_strategy_components()
    keys = {
        item.key
        for group in output.groups
        for item in group.components
    }
    assert "feature.sma.v1" in keys
    assert "sizer.default.v1" in keys
    assert "feature.python_callback.v1" not in keys
    assert all(item.json_schema for group in output.groups for item in group.components)
    assert all(item.lookback for group in output.groups for item in group.components)


def test_validate_returns_all_semantic_issues_without_running(mcp_app):
    payload = ma_cross_spec().model_dump(mode="json")
    payload["universe"] = ["XNAS:UNKNOWN"]
    payload["rules"][0]["when"]["left"]["feature"] = "missing"

    output = mcp_app.validate_strategy_spec(
        ValidateStrategySpecInput.model_validate({
            "dataset_id": "daily-us",
            "strategy_spec": payload,
        })
    )

    assert output.valid is False
    assert {issue.code for issue in output.issues} == {
        "unknown_instrument",
        "unknown_feature_reference",
    }
    assert output.spec_fingerprint is None
    assert mcp_app.test_probe.engine_run_count == 0
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_catalog_and_validation.py -q
```

- [ ] **Step 3: Implement strict request and response DTOs**

All models inherit this base:

```python
class MCPModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )
```

Define exact public models:

```python
class MCPBacktestConfig(MCPModel):
    start: datetime
    end: datetime
    timeframe: Literal["1d", "1m"]
    calendar: Literal["union", "intersection"] = "union"
    initial_cash: Decimal = Field(gt=0, max_digits=24, decimal_places=8)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)


class ValidateStrategySpecInput(MCPModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    strategy_spec: dict[str, JsonValue]
    config: MCPBacktestConfig | None = None


class ValidateStrategySpecOutput(MCPModel):
    valid: bool
    spec_fingerprint: str | None
    dataset_fingerprint: str | None
    issues: tuple[PublicValidationIssue, ...]
    required_lookback_bars: int | None


class PublicValidationIssue(MCPModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    path: tuple[str | int, ...]
    message: str = Field(min_length=1, max_length=500)


class PublicToolError(MCPModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=500)
    details: dict[str, PublicScalar] = Field(default_factory=dict)


T = TypeVar("T", bound=MCPModel)


class ToolEnvelope(MCPModel, Generic[T]):
    ok: bool
    data: T | None
    error: PublicToolError | None
```

Add a model validator requiring exactly one of `data` and `error`; `ok=True`
requires data and `ok=False` requires an error. Every MCP tool uses a
specialized envelope on the wire. Validation of an invalid StrategySpec is
still a successful tool call: its envelope has `ok=True` and
`data.valid=False`.

JSON transport necessarily decodes arrays, timestamps, and decimal numbers
before model construction, so the base does not enable Pydantic's global
`strict=True`. Field types, discriminated unions, bounds, `extra="forbid"`,
finite-number checks, and explicit model validators define accepted
coercions. Add a validator requiring `start < end` and rejecting booleans for
all integer/Decimal fields.

Import `JsonValue` from Pydantic. The two spec-taking inputs deliberately
accept only `dict[str, JsonValue]`, not a prevalidated `StrategySpec`, so
`validate_strategy_spec` can return structured schema issues instead of the
SDK rejecting the call before the application sees it. The application
immediately calls `StrategySpec.model_validate()` and retains the parsed model
only on success; JSON values cannot carry callables or Python objects.

`ComponentCatalogOutput` groups descriptors by `feature`, `condition`,
`action`, `sizer`, and `risk`. Each descriptor exposes only stable key,
description, JSON schema, units, a human-readable lookback formula, and a
short JSON example. Its key is constructed explicitly as
`"{kind.value}.{name}.v1"` from the frozen registry descriptor.
It does not expose a Python module, class, callable, source path, or registry
factory.

Convert internal exceptions to `MCPPublicError` at the application boundary.
Return all schema/semantic issues in deterministic `(path, code)` order.
Never include Pydantic input echoes in an error message.
If schema parsing fails, return all schema issues and skip semantic passes
because there is no valid typed graph to inspect. If it succeeds, return all
semantic issues from the validator.

At composition time, build one reentrant `BacktestService` for each
`DatasetBinding`, injecting that binding's source and instrument catalog.
`ConfiguredBacktestServices` copies this exact ID→service map into a
`MappingProxyType` and verifies its keys exactly equal
`DatasetCatalog.list_descriptors()`. MCP code resolves the service by the same
validated dataset ID; it never passes a caller-selected source/path into a
service.

- [ ] **Step 4: Enforce request byte bounds at both adapter and application boundaries**

Add:

```python
def validate_request_size(payload: bytes, limit: int) -> None:
    if len(payload) > limit:
        raise MCPPublicError(
            code="request_too_large",
            message="Request exceeds the configured byte limit.",
            details={"limit_bytes": limit},
        )
```

The transport-independent application canonical-serializes each parsed input
and applies the limit before resolving a dataset or component. Add an
`MCPServer` refusal middleware in Task 6 that canonical-serializes
`ctx.params` before SDK argument validation and raises a sanitized protocol
error when it exceeds the same limit. The middleware is only a quota guard;
domain validation remains in the application.

- [ ] **Step 5: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_catalog_and_validation.py tests/mcp/test_security.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/mcp
rtk git add src/pybacktest/mcp tests/mcp
rtk git commit -m "feat: expose safe MCP catalog and validation"
```

## Task 3: Bounded Local Run Lifecycle

**Files:**
- Create: `src/pybacktest/application/run_repository.py`
- Create: `src/pybacktest/adapters/runs/__init__.py`
- Create: `src/pybacktest/adapters/runs/memory.py`
- Modify: `src/pybacktest/mcp/models.py`
- Modify: `src/pybacktest/mcp/application.py`
- Create: `tests/mcp/test_run_lifecycle.py`

**Interfaces:**
- Consumes: `SpecBacktestRequest`, `BacktestService.run_spec()`,
  `BacktestResult`, `ExperimentService`, dataset metadata, component
  allowlist, and MCP quotas.
- Produces:
  - `RunStatus` enum: `queued`, `running`, `completed`, `failed`
  - immutable `RunJob`
  - `RunRepository` protocol
  - thread-safe `InMemoryRunRepository`
  - `RunBacktestInput`, `RunBacktestOutput`
  - `GetBacktestResultInput`, `GetBacktestResultOutput`
  - `MCPApplication.run_backtest()`
  - `MCPApplication.get_backtest_result()`

- [ ] **Step 1: Write failing lifecycle, revalidation, and quota tests**

```python
import pytest

from pybacktest.mcp.errors import MCPPublicError
from pybacktest.mcp.models import GetBacktestResultInput, RunBacktestInput
from tests.factories import ma_cross_spec, simulation_config


def test_run_revalidates_and_returns_a_bounded_completed_summary(mcp_app):
    accepted = mcp_app.run_backtest(
        RunBacktestInput.model_validate({
            "dataset_id": "daily-us",
            "strategy_spec": ma_cross_spec().model_dump(mode="json"),
            "config": simulation_config(),
        })
    )
    assert accepted.status.value in {"queued", "running", "completed"}

    result = mcp_app.get_backtest_result(
        GetBacktestResultInput(run_id=accepted.run_id)
    )
    assert result.status.value == "completed"
    assert result.summary is not None
    assert result.summary.run_id == accepted.run_id
    assert result.resources.manifest.endswith("/manifest")
    assert "internal_path" not in result.model_dump()
    assert mcp_app.test_probe.validation_count == 2
    assert mcp_app.test_probe.engine_run_count == 1


def test_run_rejects_before_loading_data_when_observation_quota_is_exceeded(
    quota_limited_mcp_app,
):
    request = RunBacktestInput.model_validate(
        {
            "dataset_id": "daily-us",
            "strategy_spec": ma_cross_spec().model_dump(mode="json"),
            "config": simulation_config(),
        }
    )
    with pytest.raises(MCPPublicError) as caught:
        quota_limited_mcp_app.run_backtest(request)
    assert caught.value.code == "bar_observation_limit_exceeded"
    assert quota_limited_mcp_app.test_probe.source_load_count == 0
    assert quota_limited_mcp_app.test_probe.engine_run_count == 0
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_run_lifecycle.py -q
```

- [ ] **Step 3: Implement the run repository and executor seam**

Define:

```python
class RunExecutor(Protocol):
    def submit(self, operation: Callable[[], None]) -> Future[None]: ...


class RunRepository(Protocol):
    def reserve(
        self,
        run_id: RunId,
        request_fingerprint: str,
        *,
        max_active: int,
    ) -> RunJob: ...
    def mark_running(self, run_id: RunId) -> None: ...
    def complete(
        self,
        run_id: RunId,
        summary: PublicRunSummary,
        artifact_ref: ArtifactRef,
    ) -> None: ...
    def fail(self, run_id: RunId, error: PublicRunError) -> None: ...
    def get(self, run_id: RunId) -> RunJob: ...
    def active_count(self) -> int: ...
```

`InMemoryRunRepository` protects every transition with one lock, returns
immutable job snapshots, and rejects duplicate IDs or status regression.
Only public DTOs cross the MCP boundary. `reserve()` counts active jobs and
inserts the reservation inside the same critical section, so two simultaneous
calls cannot both pass the final capacity slot.

`RunJob` stores `PublicRunSummary | None`, `ArtifactRef | None`, and
`PublicRunError | None` as mutually exclusive state-dependent fields.
`complete()` requires both the bounded summary and finalized artifact
reference. The worker releases the potentially large `BacktestResult` after
artifact persistence; the repository never accumulates full equity/event
tables in memory.

Production uses `ThreadPoolExecutor(max_workers=max_concurrent_runs)`. The
submitted closure owns the whole transition from `mark_running()` through
service execution, artifact persistence, and `complete()`/`fail()`, and
returns `None`; no `BacktestResult` escapes through the future. Tests inject
an `InlineRunExecutor` whose already-completed `Future` makes state
transitions deterministic. Reject submission with
`run_capacity_exceeded` when active count reaches the configured maximum;
do not create an unbounded executor queue.

- [ ] **Step 4: Implement `run_backtest` in this exact order**

1. Parse strict DTOs and enforce canonical request byte size.
2. Resolve `dataset_id` and its preconfigured service by exact allowlist
   lookup.
3. Parse the raw JSON object with `StrategySpec.model_validate()`; reject all
   schema issues before reading typed fields.
4. Check timeframe, period, typed universe, and instrument count against the
   descriptor.
5. Estimate instrument × bar observations from descriptor metadata and reject
   over quota before `MarketDataSource.load()`.
6. Call the same semantic validator used by
   `validate_strategy_spec`.
7. Check active-run capacity and atomically reserve a fresh server-generated
   `RunId`.
8. Build
   `SpecBacktestRequest(run_id=reserved_run_id, ...)`, and call
   `BacktestService.run_spec()`, which deliberately validates again at its own
   trust boundary. Assert that the returned result carries both the reserved
   run ID and resolved dataset fingerprint; a changed source fails with
   `dataset_fingerprint_changed`.
9. Persist through `LocalArtifactStore(max_bytes=max_artifact_bytes)`. It
   measures files in the temporary sibling directory, raises
   `artifact_limit_exceeded` before atomic publish, and removes only that
   verified temporary directory.
10. Optionally record the explicit experiment IDs/feedback supplied by the
   caller; never infer or request hidden reasoning.
11. Project the bounded summary and call
    `repository.complete(run_id, summary, artifact_ref)`, then release the
    full result; on failure store only a sanitized public error.

The exact execution inputs and public response are:

```python
class ExperimentLinkInput(MCPModel):
    experiment_id: str
    agent_session_id: str
    parent_run_id: str | None = None
    feedback: str | None = Field(default=None, max_length=4_000)


class RunBacktestInput(MCPModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    strategy_spec: dict[str, JsonValue]
    config: MCPBacktestConfig
    experiment: ExperimentLinkInput | None = None


class RunBacktestOutput(MCPModel):
    run_id: str
    status: RunStatus
    summary: PublicRunSummary | None
    resources: RunResourceLinks
    error: PublicRunError | None
```

`GetBacktestResultOutput` adds warnings and progress with progress restricted
to `0`, `1`, or `None` until the engine exposes measured progress. It never
fabricates a percentage.

- [ ] **Step 5: Add failure sanitization and concurrency tests**

Assert:

- an unknown run raises `MCPPublicError(code="run_not_found")`, which the SDK
  adapter maps to an error envelope;
- an internal exception stores `run_failed` with a stable public message, not
  exception text or traceback;
- a second active run is rejected at capacity;
- completion decrements active count exactly once;
- failed, rejected, and completed transitions are deterministic;
- `run_backtest` revalidates even after a successful earlier validation call;
- returned JSON stays below the configured response projection size.

- [ ] **Step 6: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_run_lifecycle.py tests/mcp/test_security.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/application/run_repository.py src/pybacktest/adapters/runs src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/application/run_repository.py src/pybacktest/adapters/runs src/pybacktest/mcp
rtk git add src/pybacktest/application/run_repository.py src/pybacktest/adapters/runs src/pybacktest/mcp tests/mcp
rtk git commit -m "feat: run bounded MCP backtests"
```

## Task 4: Compare Runs and Explain Recorded Trades

**Files:**
- Create: `src/pybacktest/mcp/comparison.py`
- Modify: `src/pybacktest/mcp/models.py`
- Modify: `src/pybacktest/mcp/application.py`
- Create: `tests/mcp/test_compare_and_explain.py`

**Interfaces:**
- Consumes: repository status/summary, persisted canonical run configs,
  metric metadata, experiment lineage, and persisted causal events interpreted
  by core `build_trade_explanation()`.
- Produces:
  - `CompareBacktestsInput`, `CompareBacktestsOutput`
  - `ExplainTradeInput`, `ExplainTradeOutput`
  - `MCPApplication.compare_backtests()`
  - `MCPApplication.explain_trade()`

- [ ] **Step 1: Write failing comparison and explanation tests**

```python
from pybacktest.mcp.comparison import ConfigDifference
from pybacktest.mcp.models import CompareBacktestsInput, ExplainTradeInput


RUN_A = f"run_{'a' * 32}"
RUN_B = f"run_{'b' * 32}"
ORDER_1 = f"order_{'1' * 32}"


def test_compare_uses_same_metric_definitions_and_reports_config_differences(
    completed_run_app,
):
    output = completed_run_app.compare_backtests(
        CompareBacktestsInput(
            run_ids=(RUN_A, RUN_B),
            metrics=("total_return", "sharpe", "maximum_drawdown"),
        )
    )
    assert [row.run_id for row in output.runs] == [RUN_A, RUN_B]
    assert output.metric_definitions["sharpe"].formula_id == "sharpe.v1"
    assert output.config_differences == (
        ConfigDifference(
            path="strategy_spec.features.fast.window",
            values={RUN_A: 20, RUN_B: 10},
        ),
    )


def test_explain_trade_returns_only_the_recorded_causal_trace(completed_run_app):
    output = completed_run_app.explain_trade(
        ExplainTradeInput(run_id=RUN_A, order_id=ORDER_1)
    )
    assert [entry.stage for entry in output.entries] == [
        "feature",
        "rule",
        "intent",
        "risk",
        "fill",
    ]
    assert "generated_narrative" not in output.model_dump()
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_compare_and_explain.py -q
```

- [ ] **Step 3: Implement deterministic comparison**

Accept 2 through `settings.max_compare_runs` unique completed run IDs.
Metric names are a `Literal` allowlist matching core `SummaryMetrics`; reject
unknown formulas rather than evaluating expressions. Compare only metrics with
identical formula ID, annualization, risk-free-rate, and NaN policy. Return:

- input-order run rows;
- metric definitions and values;
- canonical JSON-pointer-like config differences;
- dataset/spec fingerprints;
- parent/child lineage edges when recorded;
- warnings for incomparable metrics.

The diff routine descends only parsed canonical JSON values, sorts mapping
keys, preserves list indices, and caps differences at 200 with `truncated`.

- [ ] **Step 4: Implement trace projection**

Resolve both IDs by exact typed parsing. Reject non-completed runs and orders
that do not belong to that run. Call core `build_trade_explanation()` and map
recorded entries without adding causal claims. Cap output at
`settings.max_trace_entries`; if the trace is longer, return
`trace_limit_exceeded` instead of silently omitting causal stages.
For persisted runs, apply Parquet predicate pushdown on `order_id`, verify the
events checksum, then call core `build_trade_explanation()` on those ordered
entries.

- [ ] **Step 5: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_compare_and_explain.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/mcp
rtk git add src/pybacktest/mcp tests/mcp
rtk git commit -m "feat: compare MCP runs and explain trades"
```

## Task 5: Bounded Read-Only MCP Resources

**Files:**
- Create: `src/pybacktest/mcp/resources.py`
- Create: `src/pybacktest/mcp/artifact_reader.py`
- Modify: `src/pybacktest/mcp/models.py`
- Create: `tests/mcp/test_resources.py`

**Interfaces:**
- Consumes: public component catalog, completed run DTOs, core result tables,
  and resource projection quotas.
- Produces exactly:
  - `pybacktest://components/catalog`
  - `pybacktest://runs/{run_id}/manifest`
  - `pybacktest://runs/{run_id}/summary`
  - `pybacktest://runs/{run_id}/orders`
  - `pybacktest://runs/{run_id}/fills`
  - `pybacktest://runs/{run_id}/equity`
  - `MCPResourceApplication.read_*()` projection methods

- [ ] **Step 1: Write failing resource projection tests**

```python
import pytest

from pybacktest.mcp.errors import MCPPublicError


RUN_A = f"run_{'a' * 32}"


def test_table_resource_is_bounded_and_does_not_leak_artifact_paths(
    completed_run_app,
):
    resource = completed_run_app.resources.read_orders(RUN_A)
    assert len(resource.rows) == completed_run_app.settings.max_inline_rows
    assert resource.row_count > len(resource.rows)
    assert resource.truncated is True
    encoded = resource.model_dump_json()
    assert "/Users/" not in encoded
    assert "file://" not in encoded


def test_resources_reject_cross_run_and_malformed_identifiers(completed_run_app):
    with pytest.raises(MCPPublicError) as caught:
        completed_run_app.resources.read_summary("../run-a")
    assert caught.value.code == "invalid_run_id"
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_resources.py -q
```

- [ ] **Step 3: Implement resource projections**

`manifest` and `summary` return strict public DTOs. Table resources return:

```python
class BoundedTableResource(MCPModel):
    run_id: str
    table: Literal["orders", "fills", "equity"]
    columns: tuple[PublicColumn, ...]
    rows: tuple[dict[str, PublicScalar | None], ...]
    row_count: int
    truncated: bool
    artifact_id: str
    checksum: str
```

Use the first `max_inline_rows` in canonical event/timestamp order. Decimal,
datetime, enum, and identifier values serialize to stable strings. Never
return a local path or a directly openable host URL. The opaque `artifact_id`
and checksum let local callers verify the full artifact through a separately
authorized Python API.

Before returning any projection, canonical-serialize it and enforce the
configured response/artifact bounds. A resource never reads a caller-supplied
path; a `RunRecordResolver` checks active/current-process state in
`RunRepository` and then the persisted artifact reader by exact `RunId`.

`RunArtifactReader` receives the configured artifact root once. Given a
validated `RunId`, it derives only `<root>/<run_id>/<fixed-filename>`, resolves
the path under the root, verifies `manifest.json` against the one-line
`manifest.sha256` sidecar, verifies the selected table against the manifest
checksum, and reads bounded columns/row groups. It rejects a missing or
mismatched sidecar before trusting any manifest field. It never accepts a
filename or path from a tool/resource argument. `get_backtest_result`,
comparison, explanation, and resources fall back to this reader when a
completed run predates the current stdio process; active status remains in
`RunRepository`.

- [ ] **Step 4: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_resources.py tests/mcp/test_security.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/mcp
rtk git add src/pybacktest/mcp tests/mcp
rtk git commit -m "feat: add bounded MCP resources"
```

## Task 6: Register the Official MCP SDK v2 Server

**Files:**
- Create: `src/pybacktest/mcp/server.py`
- Create: `src/pybacktest/mcp/__main__.py`
- Modify: `src/pybacktest/mcp/__init__.py`
- Create: `tests/mcp/test_server_contract.py`
- Create: `tests/mcp/test_stdio_entrypoint.py`

**Interfaces:**
- Consumes: transport-independent application methods from Tasks 2-5.
- Produces:
  - `create_mcp_server(application: MCPApplication) -> MCPServer`
  - six MCP tools:
    `list_strategy_components`, `validate_strategy_spec`, `run_backtest`,
    `get_backtest_result`, `compare_backtests`, `explain_trade`
  - six MCP resources from Task 5
  - `main() -> None`, which starts stdio transport

- [ ] **Step 1: Write failing in-memory SDK contract tests**

```python
import pytest
from mcp import Client

from pybacktest.mcp.server import create_mcp_server


@pytest.mark.anyio
async def test_server_exposes_exact_tool_contract(mcp_app):
    server = create_mcp_server(mcp_app)
    async with Client(server) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {
            "list_strategy_components",
            "validate_strategy_spec",
            "run_backtest",
            "get_backtest_result",
            "compare_backtests",
            "explain_trade",
        }

        result = await client.call_tool("list_strategy_components", {})
        assert result.is_error is False
        assert result.structured_content["ok"] is True
        assert result.structured_content["data"]["catalog_version"] == "1"

        missing = await client.call_tool(
            "get_backtest_result",
            {"run_id": f"run_{'a' * 32}"},
        )
        assert missing.is_error is True
        assert missing.structured_content["error"]["code"] == "run_not_found"


@pytest.mark.anyio
async def test_server_exposes_exact_resource_templates(mcp_app):
    server = create_mcp_server(mcp_app)
    async with Client(server) as client:
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        assert {str(item.uri) for item in resources.resources} == {
            "pybacktest://components/catalog",
        }
        assert {item.uri_template for item in templates.resource_templates} == {
            "pybacktest://runs/{run_id}/manifest",
            "pybacktest://runs/{run_id}/summary",
            "pybacktest://runs/{run_id}/orders",
            "pybacktest://runs/{run_id}/fills",
            "pybacktest://runs/{run_id}/equity",
        }
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_server_contract.py -q
```

- [ ] **Step 3: Register typed tools with thin closures**

Use the SDK v2 API and v2 snake-case protocol fields:

```python
from typing import Annotated

from mcp.server import MCPServer
from mcp_types import CallToolResult, TextContent


CatalogEnvelope = ToolEnvelope[ComponentCatalogOutput]
ValidationEnvelope = ToolEnvelope[ValidateStrategySpecOutput]
RunEnvelope = ToolEnvelope[RunBacktestOutput]
ResultEnvelope = ToolEnvelope[GetBacktestResultOutput]
ComparisonEnvelope = ToolEnvelope[CompareBacktestsOutput]
ExplanationEnvelope = ToolEnvelope[ExplainTradeOutput]


def create_mcp_server(application: MCPApplication) -> MCPServer:
    server = MCPServer(
        "pybacktest",
        version="0.2.0",
        middleware=[request_size_middleware(application.settings)],
    )

    @server.tool()
    def list_strategy_components() -> Annotated[CallToolResult, CatalogEnvelope]:
        return as_tool_result(
            CatalogEnvelope,
            application.list_strategy_components,
        )

    @server.tool()
    def validate_strategy_spec(
        dataset_id: str,
        strategy_spec: dict[str, JsonValue],
        config: MCPBacktestConfig | None = None,
    ) -> Annotated[CallToolResult, ValidationEnvelope]:
        request = ValidateStrategySpecInput(
            dataset_id=dataset_id,
            strategy_spec=strategy_spec,
            config=config,
        )
        return as_tool_result(
            ValidationEnvelope,
            lambda: application.validate_strategy_spec(request),
        )

    @server.tool()
    def run_backtest(
        dataset_id: str,
        strategy_spec: dict[str, JsonValue],
        config: MCPBacktestConfig,
        experiment: ExperimentLinkInput | None = None,
    ) -> Annotated[CallToolResult, RunEnvelope]:
        request = RunBacktestInput(
            dataset_id=dataset_id,
            strategy_spec=strategy_spec,
            config=config,
            experiment=experiment,
        )
        return as_tool_result(
            RunEnvelope,
            lambda: application.run_backtest(request),
        )

    @server.tool()
    def get_backtest_result(
        run_id: str,
    ) -> Annotated[CallToolResult, ResultEnvelope]:
        request = GetBacktestResultInput(run_id=run_id)
        return as_tool_result(
            ResultEnvelope,
            lambda: application.get_backtest_result(request),
        )

    @server.tool()
    def compare_backtests(
        run_ids: tuple[str, ...],
        metrics: tuple[PublicMetricName, ...],
    ) -> Annotated[CallToolResult, ComparisonEnvelope]:
        request = CompareBacktestsInput(run_ids=run_ids, metrics=metrics)
        return as_tool_result(
            ComparisonEnvelope,
            lambda: application.compare_backtests(request),
        )

    @server.tool()
    def explain_trade(
        run_id: str,
        order_id: str,
    ) -> Annotated[CallToolResult, ExplanationEnvelope]:
        request = ExplainTradeInput(run_id=run_id, order_id=order_id)
        return as_tool_result(
            ExplanationEnvelope,
            lambda: application.explain_trade(request),
        )

    register_resources(server, application.resources)
    return server
```

`as_tool_result()` catches `MCPPublicError`, creates an envelope with
`ok=False`, and sets `CallToolResult.is_error=True`. Success sets
`is_error=False`. Both paths put the same envelope JSON in `content` and
`structured_content`, so the model sees a recoverable error while clients get
its stable code without parsing display text. Unexpected exceptions become
`internal_error` without exception text, request values, stack traces, or
paths; logs contain only a generated correlation ID. Canonical envelope bytes
must fit `max_response_bytes`; otherwise return the small
`response_limit_exceeded` error envelope.

`request_size_middleware()` canonical-serializes `ctx.params`, applies
`max_request_bytes`, and refuses oversized requests with an `MCPError` whose
data is `{"code": "request_too_large", "limit_bytes": ...}`. It does not
rewrite requests.

Do not put validation, filesystem access, executor logic, or engine logic in
the decorated closures. Register the component catalog as a concrete resource
and the five run paths as resource templates.

Register resources explicitly with `mime_type="application/json"` and return
`projection.model_dump(mode="json")`:

```python
@server.resource(
    "pybacktest://components/catalog",
    mime_type="application/json",
)
def component_catalog() -> dict[str, object]:
    return application.list_strategy_components().model_dump(mode="json")


@server.resource(
    "pybacktest://runs/{run_id}/manifest",
    mime_type="application/json",
)
def run_manifest(run_id: str) -> dict[str, object]:
    return resource_result(lambda: application.resources.read_manifest(run_id))
```

Repeat the second exact shape for `summary`, `orders`, `fills`, and `equity`.
`resource_result()` maps `run_not_found` to the SDK v2
`ResourceNotFoundError`; all other failures become sanitized protocol errors.

- [ ] **Step 4: Add an explicit stdio composition root**

`main()` requires `--config PATH`, loads that one operator-owned local
configuration file, constructs configured dataset adapters,
artifact/experiment stores, component allowlist, services, repository, and
executor, then calls:

```python
create_mcp_server(application).run()
```

The default is SDK stdio. Do not accept a transport selector from MCP input.
Configuration errors go to stderr before transport starts and exit non-zero.
Once MCP transport starts, logs remain on stderr so stdout contains protocol
frames only. The composition root owns the executor and closes it in a
`finally` block; graceful shutdown waits for admitted runs and rejects new
ones, while process termination cannot publish a half-written artifact because
the artifact store is atomic.

`python -m pybacktest.mcp --check-config <config>` validates configuration and
exits without opening data or starting the server. Configuration supports only
adapter types explicitly registered by the composition root; it does not
accept Python import paths. The config path is an operator CLI concern and is
never exposed as an MCP tool parameter.

The versioned TOML shape is:

```toml
schema_version = "1"
workspace_root = ".."
artifact_root = "artifacts"
experiment_database = "experiments.sqlite3"

[limits]
max_instruments = 100
max_bar_observations = 1000000
max_concurrent_runs = 2
max_request_bytes = 1000000
max_response_bytes = 1000000
max_artifact_bytes = 250000000
max_inline_rows = 200
max_compare_runs = 10
max_trace_entries = 500

[engine]
fill_model = "next_bar_open"
intrabar_policy = "conservative"

[engine.commission]
type = "per_share"
rate_per_share = "0.005"
minimum_fee = "0"

[engine.slippage]
type = "volume_share"
impact_bps = "5"

[engine.liquidity]
type = "volume_participation"
max_volume_ratio = "0.05"

[engine.borrow_cost]
type = "none"

[engine.sizer]
type = "default"

[engine.risk]
max_leverage = "1.0"
max_position_weight = "0.25"
allow_short = true

[components]
features = ["source", "sma", "ema", "lag"]
conditions = [
  "crosses_above",
  "crosses_below",
  "greater_than",
  "less_than",
  "all",
  "any",
  "not",
]
actions = [
  "target_weight",
  "target_quantity",
  "market_order",
  "limit_order",
]
sizers = ["default"]
risks = ["max_position_weight", "max_leverage", "allow_short"]

[[datasets]]
dataset_id = "daily-us"
display_name = "Bundled US daily sample"
adapter = "parquet"
location = "data/daily-us.parquet"
timeframes = ["1d"]
instruments = ["XNAS:AAPL", "XNAS:MSFT"]
available_start = "2024-01-02T00:00:00Z"
available_end = "2024-03-22T00:00:00Z"
```

Resolve `workspace_root` relative to the configuration file only after
configuration validation, then require every other configured path to remain
under the resolved root, including after resolving existing symlinks. At
server startup, the registered adapter builds a metadata-only
observation estimator and computes the dataset fingerprint; `--check-config`
validates shape and path containment but does not open the dataset.
Every `type` field above is a closed literal dispatched to an explicit
composition-root factory; unknown values and extra fields are configuration
errors, never import paths.

- [ ] **Step 5: Add entrypoint and stdout-isolation tests**

Use a temporary valid config and subprocess to assert:

- `--check-config` exits zero without loading datasets;
- an invalid adapter key exits non-zero with a stable message;
- no absolute config/data/artifact path is printed;
- importing `pybacktest.mcp.server` has no side effects;
- calling `main()` is the only operation that starts stdio;
- core-only import remains independent of `mcp`.

- [ ] **Step 6: Verify and commit**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_server_contract.py tests/mcp/test_stdio_entrypoint.py -q
rtk proxy uv run --extra mcp ruff check src/pybacktest/mcp tests/mcp
rtk proxy uv run --extra mcp ty check src/pybacktest/mcp
rtk git add src/pybacktest/mcp tests/mcp pyproject.toml uv.lock
rtk git commit -m "feat: expose pybacktest MCP v2 server"
```

## Task 7: Security, Contract, Documentation, and End-to-End Gate

**Files:**
- Modify: `tests/mcp/test_security.py`
- Create: `tests/mcp/test_contract.py`
- Create: `tests/mcp/test_end_to_end.py`
- Create: `docs/mcp.md`
- Create: `scripts/build_example_data.py`
- Create: `examples/data/daily-us.parquet`
- Create: `examples/mcp/config.toml`
- Create: `examples/mcp/ma_cross_spec.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: all MCP interfaces from Tasks 1-6.
- Produces: one documented local AI-agent workflow and the complete MCP
  acceptance/security gate.

- [ ] **Step 1: Add a complete malicious-input matrix**

Parameterize payloads containing:

```text
python
callable
import_path
source_url
data_url
filesystem_path
shell_command
environment_variable
../../escape
file:///tmp/escape
https://example.test/data.csv
```

Inject each value at every schema location that accepts an object. Assert
strict schema rejection, no source load, no engine run, and no artifact write.
Assert the boundary-specific error contract:

- `validate_strategy_spec` successfully returns `data.valid=False` with
  structured schema issues;
- `run_backtest` returns `is_error=True` with a
  `strategy_spec_invalid` envelope;
- invalid outer tool arguments are rejected by the SDK before the handler;
- oversized canonical request payloads are refused by middleware as an MCP protocol error
  with `data.code="request_too_large"`.

Also test:

- unknown component and dataset IDs;
- oversized request, universe, date range/observation estimate, artifact,
  comparison set, and trace;
- malformed and cross-run run/order IDs;
- duplicate run IDs and status races;
- internal exceptions containing a fake secret and absolute path;
- completed result/resource isolation;
- table truncation and output size;
- requests that attempt HTTP transport selection.

- [ ] **Step 2: Write the end-to-end AI workflow test**

Through the SDK in-memory `Client`:

1. list components;
2. submit an invalid spec and inspect structured issues;
3. submit a corrected spec;
4. run two child iterations against the same dataset fingerprint;
5. poll both run IDs to completion;
6. compare metrics/config/lineage;
7. choose one recorded order and explain its causal trace;
8. read summary and bounded fill/equity resources.

Assert both runs use `BacktestService.run_spec()`, manifests retain canonical
spec/dataset/compiler fingerprints, and experiment records contain no prompt
or reasoning unless explicit feedback was supplied.

- [ ] **Step 3: Verify the security and end-to-end tests fail**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp/test_security.py tests/mcp/test_end_to_end.py -q
```

- [ ] **Step 4: Close all failures without expanding the public surface**

Fix only existing DTO validation, catalog resolution, application use cases,
repository transitions, projections, or adapter error mapping. Do not add a
seventh tool, arbitrary query language, path-based artifact download, Python
execution, or remote transport to make a test convenient.

- [ ] **Step 5: Document visible behavior**

`docs/mcp.md` and README must show:

- install the server with `uv add "pybacktest[mcp]"`, and add the adapter
  extra used by the dataset, such as
  `uv add "pybacktest[mcp,parquet]"` for the bundled example;
- operator-owned config and allowlisted dataset IDs;
- `pybacktest-mcp --check-config ...`;
- `pybacktest-mcp --config ...` as the stdio launch command;
- stdio client configuration;
- the six tools and six resource URIs;
- exact StrategySpec JSON and validation error JSON;
- the `generate → validate → run → compare → revise` workflow;
- quota defaults and stable error codes;
- artifact/resource truncation behavior;
- experiment fields and the absence of hidden reasoning;
- the security boundary and the explicit lack of remote transport.

Host examples use an absolute `pybacktest-mcp` executable path and an absolute
`--config` path because MCP hosts may start in an unrelated working directory.
The TOML's `workspace_root` remains relative to that TOML file, and
dataset/artifact paths remain relative to the declared root for portable
bundles. Set `PYTHONUNBUFFERED=1` in Python stdio host examples and send logs
only to stderr.

`scripts/build_example_data.py` writes the committed Parquet example over 80
daily UTC timestamps from 2024-01-02 through 2024-03-21. AAPL closes are
`tuple(160 - i for i in range(60)) + tuple(101 + 5 * i for i in range(20))`;
MSFT closes are `tuple(200 + i for i in range(80))`. Open is the prior close
(the first open equals its close), high is `max(open, close) + 1`, low is
`min(open, close) - 1`, and volume is `10_000`. There is no randomness, and
AAPL contains one MA(20)/MA(60) cross. Running the script twice must produce
data that loads to the same core dataset fingerprint. The example
configuration uses that project-relative sample under `examples/data/`;
tests resolve it from the configuration file, not the process working
directory.

- [ ] **Step 6: Run the complete plan-3 acceptance gate**

```bash
rtk proxy uv run --extra mcp python -m pytest tests/mcp -q
rtk proxy uv run --extra mcp python -m pytest -q
rtk proxy uv run --extra mcp ruff check src tests
rtk proxy uv run --extra mcp ruff format --check src tests
rtk proxy uv run --extra mcp ty check src/pybacktest
rtk git diff --check
```

Confirm separately that the core-only boundary still works:

```bash
rtk proxy uv run --isolated --no-dev python -c "import pybacktest"
```

- [ ] **Step 7: Commit**

```bash
rtk git add src/pybacktest/mcp src/pybacktest/application/run_repository.py src/pybacktest/adapters/runs tests/mcp docs/mcp.md scripts/build_example_data.py examples/data/daily-us.parquet examples/mcp README.md pyproject.toml uv.lock
rtk git commit -m "docs: complete safe MCP agent workflow"
```

## MCP Plan Completion Check

Run:

```bash
rtk git status --short
rtk proxy uv run --extra mcp python -m pytest -q
rtk proxy uv run --extra mcp ruff check src tests
rtk proxy uv run --extra mcp ruff format --check src tests
rtk proxy uv run --extra mcp ty check src/pybacktest
rtk git diff --check
```

The worktree must be clean. The implemented boundary is:

```text
MCP SDK v2 stdio
    → strict MCP DTOs and quotas
    → allowlisted DatasetCatalog and component catalog
    → StrategySpec schema + semantic validation
    → BacktestService.run_spec()
    → the same BacktestEngine used by Python callers
    → immutable results, experiment lineage, and bounded projections
```
