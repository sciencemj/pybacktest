from decimal import Decimal

import pytest

from pybacktest import ConfigurationError, InstrumentId
from pybacktest.data.features import FeatureBuilder
from streamlit_ui.strategies import (
    BuyAndHold,
    MovingAverageCrossAll,
    RsiReversion,
    equal_weight,
    feature_name,
    rsi,
)

AAPL = InstrumentId.parse("YF:AAPL")
MSFT = InstrumentId.parse("YF:MSFT")


def test_equal_weight_is_quantized_to_six_places():
    assert equal_weight(1) == Decimal("1.000000")
    assert equal_weight(3) == Decimal("0.333333")


def test_rsi_matches_hand_computed_value():
    # Changes: +1, -0.5, +2, -1 -> gains 3, losses 1.5 -> RS 2 -> RSI 66.67.
    assert rsi([10.0, 11.0, 10.5, 12.5, 11.5]) == pytest.approx(200 / 3)


def test_rsi_edge_cases():
    assert rsi([1.0, 2.0, 3.0]) == 100.0
    assert rsi([3.0, 3.0, 3.0]) == 50.0
    assert rsi([3.0, 2.0, 1.0]) == 0.0


def test_feature_names_are_namespaced_per_instrument():
    strategy = MovingAverageCrossAll(instruments=(AAPL, MSFT), fast=2, slow=3)
    plan = strategy.build_features(FeatureBuilder())
    names = {node.name for node in plan.nodes}
    assert feature_name("fast", AAPL) in names
    assert feature_name("fast", MSFT) in names
    assert feature_name("fast", AAPL) == "fast:YF:AAPL"


def test_rsi_declares_period_lags():
    strategy = RsiReversion(instruments=(AAPL,), period=3, lower=30, upper=70)
    plan = strategy.build_features(FeatureBuilder())
    names = {node.name for node in plan.nodes}
    assert {"close:YF:AAPL", "close_lag1:YF:AAPL", "close_lag3:YF:AAPL"} <= names


@pytest.mark.parametrize(
    ("strategy", "warmup"),
    [
        (BuyAndHold(instruments=(AAPL,)), 1),
        (MovingAverageCrossAll(instruments=(AAPL,), fast=5, slow=20), 21),
        (RsiReversion(instruments=(AAPL,), period=14, lower=30, upper=70), 15),
    ],
)
def test_warmup_bars(strategy, warmup):
    assert strategy.warmup_bars == warmup


@pytest.mark.parametrize(
    "build",
    [
        lambda: MovingAverageCrossAll(instruments=(AAPL,), fast=20, slow=20),
        lambda: MovingAverageCrossAll(instruments=(AAPL,), fast=0, slow=20),
        lambda: RsiReversion(instruments=(AAPL,), period=1, lower=30, upper=70),
        lambda: RsiReversion(instruments=(AAPL,), period=31, lower=30, upper=70),
        lambda: RsiReversion(instruments=(AAPL,), period=14, lower=70, upper=30),
        lambda: BuyAndHold(instruments=()),
        lambda: BuyAndHold(instruments=(AAPL, AAPL)),
    ],
)
def test_invalid_parameters_raise_configuration_error(build):
    with pytest.raises(ConfigurationError):
        build()
