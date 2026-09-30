from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pybacktest.data.features import (
    FeatureBuilder,
    FeatureExecutor,
    FeatureNode,
    FeaturePlan,
    FeatureSet,
    FeatureView,
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
    view = (
        FeatureExecutor()
        .execute(
            builder.plan(),
            dataset,
        )
        .view(dataset.timestamps[1])
    )

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


def test_feature_view_is_opaque_and_repr_never_discloses_columns():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)
    view = features.view(dataset.timestamps[1])

    with pytest.raises(TypeError, match="FeatureSet"):
        FeatureView(_columns=features.columns, _index=1)  # type: ignore[call-arg]

    representation = repr(view)
    assert "array" not in representation
    assert "mappingproxy" not in representation
    assert str(dataset.timestamps[1]) in representation
    assert view.timestamp == dataset.timestamps[1]


def test_feature_view_has_no_positional_pattern_surface():
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    view = (
        FeatureExecutor()
        .execute(
            builder.plan(),
            dataset,
        )
        .view(dataset.timestamps[0])
    )

    def match_positionally(candidate: object) -> bool:
        match candidate:
            case FeatureView(_, _):
                return True
            case _:
                return False

    with pytest.raises(TypeError):
        match_positionally(view)


def test_sma_stays_finite_for_large_finite_observations():
    dataset, instrument = one_instrument_dataset(
        closes=[1e308, 1e308],
    )
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma2", close, window=2)

    result = FeatureExecutor().execute(builder.plan(), dataset)

    np.testing.assert_allclose(
        result.column("sma2"),
        [np.nan, 1e308],
        equal_nan=True,
    )
    assert np.isfinite(result.column("sma2")[1])


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (1, [1e308, 1e308, 1e308, 1e308]),
        (2, [np.nan, 1e308, 1e308, 1e308]),
    ],
)
def test_sma_repeated_large_values_never_overflow_prefix_accumulation(
    window: int,
    expected: list[float],
):
    dataset, instrument = one_instrument_dataset(
        closes=[1e308, 1e308, 1e308, 1e308],
    )
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma", close, window=window)

    result = FeatureExecutor().execute(builder.plan(), dataset)

    np.testing.assert_allclose(
        result.column("sma"),
        expected,
        equal_nan=True,
    )
    assert not np.isinf(result.column("sma")).any()


@pytest.mark.parametrize("window", [1, 2])
def test_sma_float64_max_is_warning_free_and_finite(window: int):
    maximum = np.finfo(np.float64).max
    dataset, instrument = one_instrument_dataset(
        closes=[maximum, maximum],
    )
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma", close, window=window)

    with np.errstate(over="raise", invalid="raise"):
        result = FeatureExecutor().execute(builder.plan(), dataset)

    column = result.column("sma")
    if window == 1:
        assert column[0] == maximum
    else:
        assert np.isnan(column[0])
    assert column[1] == maximum
    assert not np.isinf(column).any()


def test_sma_preserves_the_smallest_positive_subnormal_average():
    smallest = np.nextafter(np.float64(0), np.float64(1))
    dataset, instrument = one_instrument_dataset(
        closes=[smallest, smallest],
    )
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma2", close, window=2)

    result = FeatureExecutor().execute(builder.plan(), dataset)

    assert np.isnan(result.column("sma2")[0])
    assert result.column("sma2")[1] == smallest


def test_sma_exact_fallback_recovers_subnormals_after_a_large_prefix():
    smallest = np.nextafter(np.float64(0), np.float64(1))
    dataset, instrument = one_instrument_dataset(
        closes=[1e300, smallest, smallest],
    )
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma2", close, window=2)

    result = FeatureExecutor().execute(builder.plan(), dataset)

    assert result.column("sma2")[2] == smallest


def test_feature_set_rejects_infinite_published_values():
    with pytest.raises(ConfigurationError, match="infinite"):
        FeatureSet(
            timestamps=np.array([timestamp(0)]),
            columns={"invalid": np.array([np.inf])},
        )


def test_direct_feature_plan_rejects_wrong_lookback_recurrence():
    instrument = make_instrument().id
    source = FeatureNode(
        name="close",
        operator="source",
        inputs=(),
        instrument=instrument,
        field="close",
        parameter=None,
        lookback=0,
    )

    with pytest.raises(ConfigurationError, match="source"):
        FeatureNode(
            name="bad_source",
            operator="source",
            inputs=(),
            instrument=instrument,
            field="close",
            parameter=None,
            lookback=1,
        )

    mutations = (
        FeatureNode(
            name="bad_lag",
            operator="lag",
            inputs=("close",),
            instrument=None,
            field=None,
            parameter=2,
            lookback=1,
        ),
        FeatureNode(
            name="bad_sma",
            operator="sma",
            inputs=("close",),
            instrument=None,
            field=None,
            parameter=3,
            lookback=3,
        ),
        FeatureNode(
            name="bad_ema",
            operator="ema",
            inputs=("close",),
            instrument=None,
            field=None,
            parameter=3,
            lookback=1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(ConfigurationError, match="lookback"):
            FeaturePlan(nodes=(source, mutation))


@pytest.mark.parametrize(
    ("method", "parameter_name"),
    [
        ("lag", "periods"),
        ("sma", "window"),
        ("ema", "span"),
    ],
)
def test_feature_parameters_have_a_consistent_supported_maximum(
    method: str,
    parameter_name: str,
):
    _dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    operation = getattr(builder, method)

    operation("at_limit", close, **{parameter_name: 1_000_000})
    with pytest.raises(ConfigurationError, match="at most"):
        operation(
            "above_limit",
            close,
            **{parameter_name: 1_000_001},
        )


@pytest.mark.parametrize("parameter", [True, 1_000_001, 10**1000])
def test_direct_feature_nodes_reject_unsupported_parameters(
    parameter: object,
):
    with pytest.raises(ConfigurationError, match="parameter"):
        FeatureNode(
            name="invalid",
            operator="ema",
            inputs=("close",),
            instrument=None,
            field=None,
            parameter=parameter,  # type: ignore[arg-type]
            lookback=0,
        )


def test_huge_ema_span_raises_configuration_error_not_overflow():
    _dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")

    with pytest.raises(ConfigurationError, match="at most"):
        builder.ema("huge", close, span=10**1000)


@pytest.mark.parametrize(
    "not_a_scalar",
    [
        np.array([timestamp(0)]),
        np.array(timestamp(0)),
    ],
)
def test_feature_set_view_rejects_datetime_arrays(
    not_a_scalar: object,
):
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)

    with pytest.raises(ConfigurationError, match="scalar"):
        features.view(not_a_scalar)  # type: ignore[arg-type]
