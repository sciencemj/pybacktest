from dataclasses import dataclass

import pytest

from pybacktest.application.requests import BacktestRequest
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.errors import LookaheadViolation
from pybacktest.domain.market import MarketSlice
from pybacktest.ports.strategy import StrategyContext

from .test_engine import _engine, _simulation, _two_bar_dataset


@dataclass(frozen=True)
class _FutureFeatureStrategy:
    instrument: object

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        builder.source("close", self.instrument, "close")
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple:
        del market
        context.features.at("close", offset=1)
        return ()


def test_strategy_cannot_read_a_future_feature_value() -> None:
    dataset = _two_bar_dataset()
    strategy = _FutureFeatureStrategy(next(iter(dataset.instruments)))

    with pytest.raises(LookaheadViolation):
        _engine(dataset).run(
            BacktestRequest(
                strategy=strategy,
                simulation=_simulation(dataset),
            )
        )
