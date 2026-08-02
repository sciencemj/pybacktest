"""Assemble immutable run manifests from explicit simulation inputs."""

from __future__ import annotations

from datetime import UTC

import numpy as np

from pybacktest.application.provenance import ProvenanceDescriptor
from pybacktest.application.requests import SimulationRequest
from pybacktest.data.dataset import MarketDataSet
from pybacktest.data.features import FeaturePlan
from pybacktest.domain.identifiers import RunId
from pybacktest.engine._fingerprint import (
    _component_fingerprint,
    _fingerprint,
    _version_identity,
)
from pybacktest.engine._time import _as_datetime
from pybacktest.results.models import FrozenMapping, RunManifest

_LIBRARY_VERSION = "0.2.0"
_SCHEMA_VERSION = "results.v1"


def _manifest(
    *,
    simulation: SimulationRequest,
    feature_plan: FeaturePlan,
    dataset: MarketDataSet,
    run_id: RunId,
    provenance: ProvenanceDescriptor,
    data_source: object,
    broker_factory: object,
    order_sizer: object,
    risk_policy: object,
    calendar: np.ndarray,
) -> RunManifest:
    """Build the canonical manifest without reaching into session state."""
    return RunManifest(
        run_id=run_id,
        library_version=_LIBRARY_VERSION,
        schema_version=_SCHEMA_VERSION,
        canonical_request=FrozenMapping.from_mapping(
            {
                "universe": [str(item) for item in simulation.universe],
                "period": {
                    "start": simulation.period.start.astimezone(UTC).isoformat(),
                    "end": simulation.period.end.astimezone(UTC).isoformat(),
                },
                "timeframe": {
                    "unit": simulation.timeframe.unit.value,
                    "count": simulation.timeframe.count,
                },
                "calendar": {
                    "mode": simulation.calendar.mode.value,
                    "max_staleness_bars": simulation.calendar.max_staleness_bars,
                },
                "initial_cash": {
                    "amount": str(simulation.initial_cash.amount),
                    "currency": simulation.initial_cash.currency,
                },
                "seed": simulation.seed,
                "metrics": {
                    "risk_free_rate": str(simulation.metrics.risk_free_rate),
                    "annualization_periods": simulation.metrics.annualization_periods,
                },
                "feature_plan_fingerprint": _fingerprint(feature_plan),
                "execution_fingerprint": _component_fingerprint(
                    broker_factory,
                    order_sizer,
                    risk_policy,
                ),
                "provenance": provenance.canonical_details(),
            }
        ),
        strategy_identity=provenance.strategy_identity,
        strategy_fingerprint=provenance.strategy_fingerprint,
        spec_identity=provenance.spec_identity,
        compiler_identity=provenance.compiler_identity,
        dataset_fingerprint=dataset.fingerprint,
        seed=simulation.seed,
        adapter_versions=FrozenMapping.from_mapping(
            {
                "data": _version_identity(data_source),
                "broker": _version_identity(broker_factory),
            }
        ),
        model_versions=FrozenMapping.from_mapping(
            {
                "order_sizer": _version_identity(order_sizer),
                "risk_policy": _version_identity(risk_policy),
            }
        ),
        started_at=_as_datetime(calendar[0]),
        ended_at=_as_datetime(calendar[-1]),
    )
