from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pybacktest.data.features import (
    FeatureBuilder,
    FeatureExecutor,
    FeatureNode,
    FeaturePlan,
)
from pybacktest.domain.errors import ConfigurationError, LookaheadViolation
from tests.factories import (
    bar_series,
    market_dataset,
    one_instrument_dataset,
    timestamp,
)
from tests.factories import instrument as make_instrument


def test_sma_is_vectorized_and_prefix_invariant():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3, 4, 5])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma3", close, window=3)
    plan = builder.plan()

    full = FeatureExecutor().execute(plan, dataset)
    prefix = FeatureExecutor().execute(plan, dataset.prefix(3))

    np.testing.assert_allclose(full.column("sma3")[:3], prefix.column("sma3"))
    np.testing.assert_allclose(
        full.column("sma3"),
        [np.nan, np.nan, 2, 3, 4],
    )


def test_negative_lag_and_duplicate_feature_names_are_rejected():
    builder = FeatureBuilder()
    _dataset, instrument = one_instrument_dataset(closes=[1, 2])
    close = builder.source("close", instrument, "close")
    with pytest.raises(ConfigurationError, match="positive"):
        builder.lag("future", close, periods=-1)
    with pytest.raises(ConfigurationError, match="duplicate"):
        builder.source("close", instrument, "open")


def test_feature_view_rejects_future_timestamp():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)
    view = features.view(dataset.timestamps[1])
    with pytest.raises(LookaheadViolation):
        view.at("close", offset=1)


def test_ema_uses_adjust_false_recurrence_and_is_prefix_invariant():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3, 4])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    lagged = builder.lag("previous_close", close, periods=1)
    builder.ema("ema3", lagged, span=3)
    plan = builder.plan()

    full = FeatureExecutor().execute(plan, dataset)
    prefix = FeatureExecutor().execute(plan, dataset.prefix(3))

    np.testing.assert_allclose(
        full.column("ema3"),
        [np.nan, 1.0, 1.5, 2.25],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        full.column("ema3")[:3],
        prefix.column("ema3"),
        equal_nan=True,
    )


def test_features_roll_on_their_own_instrument_observation_clock():
    aapl = make_instrument("AAPL")
    msft = make_instrument("MSFT")
    dataset = market_dataset(
        {
            aapl: bar_series(
                closes=[1, 3, 5],
                timestamps=[timestamp(0), timestamp(2), timestamp(4)],
            ),
            msft: bar_series(
                closes=[10, 20],
                timestamps=[timestamp(1), timestamp(3)],
            ),
        }
    )
    builder = FeatureBuilder()
    aapl_close = builder.source("aapl_close", aapl.id, "close")
    builder.sma("aapl_sma2", aapl_close, window=2)
    builder.source("msft_close", msft.id, "close")

    features = FeatureExecutor().execute(builder.plan(), dataset)

    np.testing.assert_allclose(
        features.column("aapl_close"),
        [1, np.nan, 3, np.nan, 5],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        features.column("aapl_sma2"),
        [np.nan, np.nan, 2, np.nan, 4],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        features.column("msft_close"),
        [np.nan, 10, np.nan, 20, np.nan],
        equal_nan=True,
    )


def test_builder_rejects_invalid_fields_parameters_and_foreign_nodes():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    del dataset
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")

    with pytest.raises(ConfigurationError, match="unknown source field"):
        builder.source("invalid", instrument, "adjusted_close")
    with pytest.raises(ConfigurationError, match="positive"):
        builder.lag("lag_zero", close, periods=0)
    with pytest.raises(ConfigurationError, match="positive"):
        builder.sma("sma_zero", close, window=0)
    with pytest.raises(ConfigurationError, match="positive"):
        builder.ema("ema_bool", close, span=True)  # type: ignore[arg-type]

    other = FeatureBuilder()
    with pytest.raises(ConfigurationError, match="different builder"):
        other.sma("foreign", close, window=2)


def test_plan_is_topological_and_nodes_are_frozen():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    del dataset
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    lag = builder.lag("lag1", close, periods=1)
    builder.sma("smoothed", lag, window=2)

    plan = builder.plan()

    assert isinstance(plan.nodes, tuple)
    assert tuple(node.name for node in plan.nodes) == (
        "close",
        "lag1",
        "smoothed",
    )
    with pytest.raises(FrozenInstanceError):
        plan.nodes[0].name = "changed"  # type: ignore[misc]


def test_published_feature_arrays_and_mappings_are_deeply_immutable():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)

    assert not features.timestamps.flags.writeable
    assert not features.column("close").flags.writeable
    with pytest.raises(ValueError):
        features.column("close").setflags(write=True)
    with pytest.raises(TypeError):
        features.columns["close"] = np.array([9.0])  # type: ignore[index]


def test_feature_view_negative_offsets_remain_within_pinned_clock():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)

    view = features.view(dataset.timestamps[1])

    assert view.at("close") == 2.0
    assert view.at("close", offset=-1) == 1.0
    with pytest.raises(ConfigurationError, match="precedes"):
        view.at("close", offset=-2)
    with pytest.raises(LookaheadViolation):
        view.at("close", offset=10)


def test_feature_view_does_not_publish_full_future_columns():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    view = FeatureExecutor().execute(
        builder.plan(),
        dataset,
    ).view(dataset.timestamps[1])

    with pytest.raises(AttributeError):
        _ = view.columns


def test_lag_never_backfills_future_values():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.lag("lag2", close, periods=2)

    features = FeatureExecutor().execute(builder.plan(), dataset)

    np.testing.assert_allclose(
        features.column("lag2"),
        [np.nan, np.nan, 1],
        equal_nan=True,
    )


def test_feature_plan_copies_nodes_and_rejects_non_topological_graphs():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    del dataset
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    lagged = builder.lag("lagged", close, periods=1)

    source_nodes = [close, lagged]
    plan = FeaturePlan(nodes=source_nodes)  # type: ignore[arg-type]
    source_nodes.clear()

    assert tuple(node.name for node in plan.nodes) == ("close", "lagged")
    with pytest.raises(ConfigurationError, match="topological"):
        FeaturePlan(nodes=(lagged, close))


def test_feature_node_rejects_operators_outside_the_closed_graph():
    with pytest.raises(ConfigurationError, match="operator"):
        FeatureNode(
            name="generic",
            operator="callable",  # type: ignore[arg-type]
            inputs=(),
            instrument=None,
            field=None,
            parameter=None,
            lookback=0,
        )
