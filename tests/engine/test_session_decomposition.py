"""Compatibility contracts for decomposing the simulation session."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.util import resolve_name
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import pybacktest
import pybacktest.engine as engine_api
import pybacktest.engine.session as session_api
from pybacktest.application.requests import BacktestRequest
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.money import Quantity
from pybacktest.engine._boundary import _FixedDatasetSessionBoundary
from pybacktest.engine._broker_guard import _BrokerCallOrigin, _engine_active_orders
from pybacktest.engine._fingerprint import _fingerprint
from pybacktest.engine._identity import _derive_fill_id, _RunIdSequence
from pybacktest.engine._loading import _load_dataset
from pybacktest.engine._manifest import _LIBRARY_VERSION, _SCHEMA_VERSION
from pybacktest.engine._time import (
    _as_datetime,
    _as_np_datetime,
    _validate_calendar_precision,
)
from pybacktest.results.models import BacktestResult
from tests.engine.test_engine import (
    _BuyWhenFlat,
    _engine,
    _simulation,
    _StaticSource,
    _two_bar_dataset,
)
from tests.integration.test_golden_scenario import _golden_result

_ROOT_ALL = [
    "AccountingInvariantError",
    "AdapterContractError",
    "ArtifactDurabilityError",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactRef",
    "BacktestEngine",
    "BacktestRequest",
    "BacktestResult",
    "BacktestService",
    "BarSeries",
    "CalendarMode",
    "CalendarPolicy",
    "CancelOrderIntent",
    "CausalStage",
    "ClockRegressionError",
    "ComponentDescriptor",
    "ConfigurationError",
    "DataValidationError",
    "DateRange",
    "DecisionReason",
    "DefaultOrderSizer",
    "DeterministicComponent",
    "EngineEvent",
    "EngineEventCode",
    "FeatureBuilder",
    "FeaturePlan",
    "FeatureView",
    "FillId",
    "Instrument",
    "InstrumentId",
    "IntrabarPolicy",
    "LimitOrderIntent",
    "LocalArtifactStore",
    "LongShortRisk",
    "LookaheadViolation",
    "MarketDataSet",
    "MarketOrderIntent",
    "MetricMetadata",
    "MetricName",
    "MetricResult",
    "MetricsConfig",
    "MissingPolicy",
    "Money",
    "MovingAverageCross",
    "NextBarOpenFill",
    "NoBorrowCost",
    "NoCommission",
    "NoLiquidityLimit",
    "NoSlippage",
    "Observation",
    "OrderId",
    "OrderIntent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PerShareCommission",
    "ProvenanceDescriptor",
    "PybacktestError",
    "Quantity",
    "ResultValidationError",
    "RiskStatus",
    "RunId",
    "RunManifest",
    "RunWarning",
    "SerializationError",
    "SimulatedBroker",
    "SimulatedBrokerFactory",
    "SimulationRequest",
    "SimulationSession",
    "StepResult",
    "Strategy",
    "StrategyContext",
    "SummaryMetrics",
    "TargetQuantity",
    "TargetWeight",
    "TimeInForce",
    "Timeframe",
    "TimeframeUnit",
    "TradeExplanation",
    "UnknownTradeError",
    "VolumeParticipationLimit",
    "VolumeShareSlippage",
    "WarningCode",
    "external_action_provenance",
    "python_strategy_provenance",
]

_ENGINE_ALL = [
    "BacktestEngine",
    "BrokerFactory",
    "Observation",
    "PortfolioLedger",
    "RecorderStateError",
    "RunRecorder",
    "SessionStateError",
    "SimulationSession",
    "StepResult",
]

_SESSION_ALL = [
    "BrokerFactory",
    "Observation",
    "SessionStateError",
    "SimulationSession",
    "StepResult",
]

_OPTIONAL_MODULES = (
    "gym",
    "gymnasium",
    "matplotlib",
    "mcp",
    "pandas",
    "pyarrow",
    "stable_baselines3",
    "streamlit",
    "yfinance",
)

_HELPER_MODULES = (
    "_time",
    "_identity",
    "_loading",
    "_boundary",
    "_fingerprint",
    "_manifest",
    "_broker_guard",
)

_ALLOWED_HELPER_EDGES = {
    "_time": set(),
    "_identity": set(),
    "_loading": {"_time"},
    "_boundary": {"_time"},
    "_fingerprint": set(),
    "_manifest": {"_fingerprint", "_time"},
    "_broker_guard": set(),
}


def _classify_helper_imports(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return forbidden imports and private-helper edges found in an AST."""
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name(
                    "." * node.level + module,
                    "pybacktest.engine",
                )
            if node.module is None:
                imported_modules.extend(
                    module if alias.name == "*" else f"{module}.{alias.name}"
                    for alias in node.names
                )
            else:
                imported_modules.append(module)

    forbidden = {
        module
        for module in imported_modules
        if module
        in {
            "pybacktest",
            "pybacktest.engine",
            "pybacktest.engine.session",
        }
        or module.startswith("pybacktest.engine.session.")
    }
    private_edges = {
        module.removeprefix("pybacktest.engine.").split(".", 1)[0]
        for module in imported_modules
        if module.startswith("pybacktest.engine._")
    }
    return forbidden, private_edges


def test_literal_public_exports_remain_unchanged() -> None:
    assert pybacktest.__all__ == _ROOT_ALL
    assert engine_api.__all__ == _ENGINE_ALL
    assert session_api.__all__ == _SESSION_ALL


def test_public_import_identities_remain_unchanged() -> None:
    assert pybacktest.Observation is engine_api.Observation
    assert engine_api.Observation is session_api.Observation
    assert pybacktest.SimulationSession is engine_api.SimulationSession
    assert engine_api.SimulationSession is session_api.SimulationSession
    assert pybacktest.StepResult is engine_api.StepResult
    assert engine_api.StepResult is session_api.StepResult
    assert engine_api.BrokerFactory is session_api.BrokerFactory
    assert engine_api.SessionStateError is session_api.SessionStateError
    assert session_api._as_datetime is _as_datetime
    assert session_api._FixedDatasetSessionBoundary is _FixedDatasetSessionBoundary
    assert _BrokerCallOrigin.PROCESS.value == "process"
    assert _engine_active_orders({}) == ()


def _fresh_import_report(module_name: str) -> tuple[int, str, list[str]]:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    probe = f"""
import importlib
import json
import sys
importlib.import_module({module_name!r})
print(json.dumps(sorted(name for name in {_OPTIONAL_MODULES!r} if name in sys.modules)))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        cwd=source_root.parent,
        env=environment,
        text=True,
    )

    optional = json.loads(completed.stdout) if completed.returncode == 0 else []
    return completed.returncode, completed.stderr, optional


def test_public_package_import_loads_no_optional_dependencies() -> None:
    returncode, stderr, optional = _fresh_import_report("pybacktest")

    assert returncode == 0, stderr
    assert optional == []


@pytest.mark.parametrize("helper_name", _HELPER_MODULES)
def test_private_helper_import_loads_no_optional_dependencies(
    helper_name: str,
) -> None:
    returncode, stderr, optional = _fresh_import_report(
        f"pybacktest.engine.{helper_name}"
    )

    assert returncode == 0, stderr
    assert optional == []


@pytest.mark.parametrize("helper_name", _HELPER_MODULES)
def test_private_helper_import_dag_is_one_way(helper_name: str) -> None:
    helper_path = Path(session_api.__file__).resolve().parent / f"{helper_name}.py"
    tree = ast.parse(helper_path.read_text("utf-8"), filename=str(helper_path))
    forbidden, private_edges = _classify_helper_imports(tree)

    assert forbidden == set()
    assert private_edges == _ALLOWED_HELPER_EDGES[helper_name]


@pytest.mark.parametrize(
    "source,expected_forbidden,expected_private_edges",
    [
        (
            "from .session import SimulationSession",
            {"pybacktest.engine.session"},
            set(),
        ),
        (
            "from . import session",
            {"pybacktest.engine.session"},
            set(),
        ),
        (
            "from ._identity import _RunIdSequence",
            set(),
            {"_identity"},
        ),
        (
            "from . import _time",
            set(),
            {"_time"},
        ),
    ],
)
def test_relative_imports_cannot_bypass_helper_graph_classification(
    source: str,
    expected_forbidden: set[str],
    expected_private_edges: set[str],
) -> None:
    forbidden, private_edges = _classify_helper_imports(ast.parse(source))

    assert forbidden == expected_forbidden
    assert private_edges == expected_private_edges


def test_time_helpers_preserve_utc_and_nanosecond_precision_policy() -> None:
    timestamp = np.datetime64("2024-01-02T14:30:00.123456", "ns")

    assert _as_datetime(timestamp) == datetime(
        2024, 1, 2, 14, 30, 0, 123456, tzinfo=UTC
    )
    assert _as_np_datetime(datetime(2024, 1, 2, 23, 30, tzinfo=UTC)) == (
        np.datetime64("2024-01-02T23:30:00", "ns")
    )
    _validate_calendar_precision(np.asarray([timestamp], dtype="datetime64[ns]"))


def test_time_helper_rejects_sub_microsecond_calendar_values() -> None:
    calendar = np.asarray(
        ["2024-01-02T14:30:00.000000001"],
        dtype="datetime64[ns]",
    )

    with pytest.raises(pybacktest.DataValidationError) as raised:
        _validate_calendar_precision(calendar)

    assert raised.value.code == "timestamp_precision_loss"


def test_dataset_loading_returns_the_validated_adapter_value_once() -> None:
    dataset = _two_bar_dataset()
    source = _StaticSource(dataset)

    assert _load_dataset(source, _simulation(dataset)) is dataset
    assert source.loads == 1


def test_run_scoped_identity_sequence_is_bit_exact() -> None:
    sequence = _RunIdSequence(RunId.parse("run_" + "6" * 32))

    assert str(sequence.next_order_id()) == ("order_dbe840e8d0af55d5b1c2b29a55d8a4be")
    assert str(sequence.next_order_id()) == ("order_56a47c3f39415206b9a71f28b7aa5f45")
    assert str(sequence.fill_id(0)) == "fill_220d2a55f29e5ee3ad484e3eb32b59ec"
    assert str(sequence.fill_id(1)) == "fill_a6c958e0a0b458c38b881b619c89997a"
    assert session_api._RunIdSequence is _RunIdSequence


@pytest.mark.parametrize("sequence", [-1, True])
def test_invalid_fill_sequence_has_stable_error_code(sequence: object) -> None:
    with pytest.raises(ConfigurationError) as raised:
        _derive_fill_id(UUID("66666666-6666-6666-6666-666666666666"), sequence)  # type: ignore[arg-type]

    assert raised.value.code == "invalid_fill_sequence"


def test_fixed_run_order_fill_and_replay_identities_are_bit_exact() -> None:
    dataset = _two_bar_dataset()
    result = _engine(dataset).run(
        BacktestRequest(
            strategy=_BuyWhenFlat(Quantity.of("1")),
            simulation=_simulation(dataset),
            run_id=RunId.parse("run_" + "1" * 32),
        )
    )

    assert str(result.orders[0].id) == "order_e12d3f42b20b542faeac1258db0665af"
    assert str(result.fills[0].id) == "fill_d7ca94e3c6a951cd8e06303df429fbd8"
    assert result.replay_fingerprint() == (
        "164d9d47ed0348bc8d9ac6627ee5ab869cb453dea8beac673c86e399f62d899a"
    )


@pytest.fixture(scope="module")
def golden_result() -> BacktestResult:
    return _golden_result()


def test_golden_fingerprints_remain_bit_exact(
    golden_result: BacktestResult,
) -> None:
    manifest = golden_result.manifest
    canonical = manifest.canonical_request

    assert golden_result.replay_fingerprint() == (
        "fe3dc2bd7b3464f4ee2fe11dd10eaf14740f0da730f808aa4c3a438537be8e3f"
    )
    assert manifest.strategy_fingerprint == (
        "0f1df0092b69c5fe51f0172290e1def1470c3fe43c883e557c642b097a73b5bd"
    )
    assert manifest.dataset_fingerprint == (
        "6a511105bbe40efae5ff6af913e0f23d6aa0586cf684a19a2c295cde936f2f01"
    )
    assert canonical["feature_plan_fingerprint"] == (
        "929c7629d9cf7a54c657d9e92e7797e95f459144616b3248666770000bc56c1d"
    )
    assert canonical["execution_fingerprint"] == (
        "b8a568a6e308a48ef7e40c9a8a217b88139e533b30e9c22771fb9b1bb799cafe"
    )
    assert _fingerprint({"value": "frozen"}) == (
        "1a350788e99ad460a7a64cc0fd4ab5355a5947d217123f05e662e1434f73f196"
    )


def test_golden_manifest_fields_and_canonical_order_remain_exact(
    golden_result: BacktestResult,
) -> None:
    manifest = golden_result.manifest
    canonical = manifest.canonical_request

    assert manifest.library_version == _LIBRARY_VERSION == "0.2.0"
    assert manifest.schema_version == _SCHEMA_VERSION == "results.v1"
    assert manifest.strategy_identity == (
        "pybacktest.strategy.components.MovingAverageCross"
    )
    assert manifest.spec_identity == "python.strategy"
    assert manifest.compiler_identity == "pybacktest.session.python.v1"
    assert manifest.seed == 7
    assert manifest.started_at == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert manifest.ended_at == datetime(2024, 1, 8, 14, 30, tzinfo=UTC)
    assert dict(manifest.adapter_versions) == {
        "data": "tests.integration.test_golden_scenario._FixedSource",
        "broker": "pybacktest.adapters.broker.simulated.SimulatedBrokerFactory",
    }
    assert dict(manifest.model_versions) == {
        "order_sizer": (
            "tests.integration.test_golden_scenario._GoodTilCancelledSizer"
        ),
        "risk_policy": "pybacktest.risk.policies.LongShortRisk",
    }
    assert tuple(canonical) == (
        "universe",
        "period",
        "timeframe",
        "calendar",
        "initial_cash",
        "seed",
        "metrics",
        "feature_plan_fingerprint",
        "execution_fingerprint",
        "provenance",
    )
    assert canonical["universe"] == ("XNAS:AAPL",)
    assert dict(canonical["period"]) == {
        "start": "2024-01-01T00:00:00+00:00",
        "end": "2024-01-09T00:00:00+00:00",
    }
    assert dict(canonical["timeframe"]) == {"unit": "day", "count": 1}
    assert dict(canonical["calendar"]) == {
        "mode": "union",
        "max_staleness_bars": 0,
    }
    assert dict(canonical["initial_cash"]) == {
        "amount": "10000",
        "currency": "USD",
    }
    assert canonical["seed"] == 7
    assert dict(canonical["metrics"]) == {
        "risk_free_rate": "0",
        "annualization_periods": 252,
    }
    assert dict(canonical["provenance"]) == {
        "spec_fingerprint": "",
        "schema_fingerprint": "",
    }
