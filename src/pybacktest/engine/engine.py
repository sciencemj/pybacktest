"""Top-level deterministic backtest orchestration."""

from dataclasses import dataclass

from pybacktest.application.requests import BacktestRequest, SimulationRequest
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.money import Quantity
from pybacktest.domain.orders import Order
from pybacktest.ports.data import MarketDataSource
from pybacktest.ports.risk import (
    OrderSizer,
    RiskContext,
    RiskDecision,
    RiskPolicy,
    RiskStatus,
)
from pybacktest.results.models import BacktestResult
from pybacktest.risk.sizing import DefaultOrderSizer

from .session import BrokerFactory, SimulationSession


@dataclass(frozen=True, slots=True)
class _AllowAllRisk:
    def evaluate(
        self,
        order: Order,
        context: RiskContext,
    ) -> RiskDecision:
        del context
        return RiskDecision(
            status=RiskStatus.PASSED,
            original_quantity=order.quantity,
            final_quantity=Quantity.of(order.quantity.value),
            codes=(),
            message="Order passed the default unrestricted risk policy.",
        )


class BacktestEngine:
    """Create isolated sessions and drive Python strategies through them."""

    def __init__(
        self,
        *,
        data_source: MarketDataSource,
        broker_factory: BrokerFactory,
        order_sizer: OrderSizer | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> None:
        if not callable(getattr(data_source, "load", None)):
            raise ConfigurationError(
                "data_source must expose a callable load method."
            )
        if not callable(getattr(broker_factory, "create", None)):
            raise ConfigurationError(
                "broker_factory must expose a callable create method."
            )
        resolved_sizer = (
            DefaultOrderSizer()
            if order_sizer is None
            else order_sizer
        )
        resolved_risk = (
            _AllowAllRisk()
            if risk_policy is None
            else risk_policy
        )
        if not isinstance(resolved_sizer, OrderSizer):
            raise ConfigurationError(
                "order_sizer must implement OrderSizer."
            )
        if not isinstance(resolved_risk, RiskPolicy):
            raise ConfigurationError(
                "risk_policy must implement RiskPolicy."
            )
        self._data_source = data_source
        self._broker_factory = broker_factory
        self._order_sizer = resolved_sizer
        self._risk_policy = resolved_risk

    def create_session(
        self,
        simulation: SimulationRequest,
        *,
        feature_plan: FeaturePlan,
        run_id: RunId | None = None,
        order_sizer: OrderSizer | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> SimulationSession:
        """Create one I/O-free, single-use isolated session."""
        if not isinstance(simulation, SimulationRequest):
            raise ConfigurationError(
                "simulation must be a SimulationRequest."
            )
        if not isinstance(feature_plan, FeaturePlan):
            raise ConfigurationError(
                "feature_plan must be a FeaturePlan."
            )
        if run_id is not None and not isinstance(run_id, RunId):
            raise ConfigurationError("run_id must be a RunId when provided.")
        resolved_sizer = (
            self._order_sizer
            if order_sizer is None
            else order_sizer
        )
        resolved_risk = (
            self._risk_policy
            if risk_policy is None
            else risk_policy
        )
        if not isinstance(resolved_sizer, OrderSizer):
            raise ConfigurationError(
                "order_sizer override must implement OrderSizer."
            )
        if not isinstance(resolved_risk, RiskPolicy):
            raise ConfigurationError(
                "risk_policy override must implement RiskPolicy."
            )
        return SimulationSession(
            simulation=simulation,
            feature_plan=feature_plan,
            run_id=run_id or RunId.new(),
            data_source=self._data_source,
            broker_factory=self._broker_factory,
            order_sizer=resolved_sizer,
            risk_policy=resolved_risk,
        )

    def run(
        self,
        request: BacktestRequest,
        *,
        order_sizer: OrderSizer | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> BacktestResult:
        """Drive exactly the public session reset/context/advance loop."""
        if not isinstance(request, BacktestRequest):
            raise ConfigurationError(
                "request must be a BacktestRequest."
            )
        builder = FeatureBuilder()
        feature_plan = request.strategy.build_features(builder)
        if not isinstance(feature_plan, FeaturePlan):
            raise ConfigurationError(
                "strategy.build_features() must return FeaturePlan."
            )
        session = self.create_session(
            request.simulation,
            feature_plan=feature_plan,
            run_id=request.run_id,
            order_sizer=order_sizer,
            risk_policy=risk_policy,
        )
        session._bind_python_strategy(
            strategy=request.strategy,
            feature_plan=feature_plan,
        )
        observation = session.reset()
        while not session.done:
            context = session.strategy_context(observation)
            intents = request.strategy.on_bar(
                context,
                observation.market,
            )
            step = session.advance(intents)
            if step.observation is not None:
                observation = step.observation
        return session.result()


__all__ = ["BacktestEngine"]
