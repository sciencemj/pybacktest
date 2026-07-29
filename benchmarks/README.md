# Performance gate

`test_engine_benchmark.py` is the measured budget for one million engine
observations. It is not collected by the default suite: `testpaths = ["tests"]`
in `pyproject.toml` excludes this directory, so the unit suite stays fast and
the gate is only ever run on purpose.

## Running it

```bash
uv run python -m pytest -m performance benchmarks/test_engine_benchmark.py -q
```

Run it in **its own process**. `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a
high-water mark for the whole process, so a peak reached by an unrelated test
earlier in the same interpreter would be attributed to the engine. Adding `-s`
prints the measured line the numbers below were taken from.

`ru_maxrss` is reported in bytes on Darwin and in kibibytes on Linux.
`benchmarks/_rss.py::normalized_peak_rss_bytes` converts both to bytes and
rejects any other platform with a stable coded error; it is unit-tested by
`tests/integration/test_rss_normalization.py`, which does run by default.

## What is measured

| | |
| --- | --- |
| Instruments | 100 (`XNAS:S0000` … `XNAS:S0099`) |
| Bars per instrument | 10,000 daily bars on one shared union clock |
| Observations | 1,000,000 |
| Features | exactly two rolling SMAs (windows 10 and 50) over one close source; no lag nodes |
| Orders | 19 sparse `MarketOrderIntent` buys, one every 500 bars, rotating instruments |
| Calendar | `CalendarPolicy.union(max_staleness_bars=0)` |
| Execution | `NextBarOpenFill(CONSERVATIVE)`, `NoCommission`, `NoSlippage`, `NoLiquidityLimit`, `NoBorrowCost` |
| Risk | `LongShortRisk(max_leverage=2, max_position_weight=None, allow_short=False)` |
| Initial cash | USD 10,000,000 |

Synthetic closes are a seeded (`default_rng(11)`) log random walk **rounded to
two decimals**, matching the instruments' `tick_size` of `0.01`. That rounding
is load-bearing: the risk policy rejects a mark that is not tick-aligned, so
unrounded prices would make every order rejected and the benchmark would time a
path with no fills at all. The test asserts `len(result.fills) > 0` and
`len(result.orders) == 19` so that regression can never pass silently.

### Timed region

Only `engine.run(request)` is timed, with `time.perf_counter()` taken
immediately before and after. Explicitly **outside** the timed region:

- building the `MarketDataSet` (array generation, validation, and the dataset
  fingerprint) — roughly 4 s on this machine;
- constructing the engine, broker factory, risk policy, and request;
- artifact serialization (`LocalArtifactStore.write()`) — never called here;
- replay fingerprinting (`result.replay_fingerprint()`) — never called here.

`pytest-benchmark`'s fixture is deliberately not used: its repeat semantics
would run `engine.run()` many times and cannot be compared against a single
wall-clock budget.

## Budget

```python
assert len(result.fills) > 0
assert elapsed_seconds <= 30.0
assert peak_rss_bytes <= 1_500_000_000
```

## Reference measurement

Re-measured for this commit on the machine that ships it.

| | |
| --- | --- |
| CPU | Apple M5, 10 cores (arm64) |
| Memory | 16 GiB (17,179,869,184 bytes) |
| OS | macOS 26.5.2 (build 25F84), `platform.system() == "Darwin"` |
| Python | 3.13.11 (main, Jan 27 2026, 23:31:20) [Clang 21.1.4] |
| NumPy | 2.4.1 |
| Command | `uv run python -m pytest -m performance benchmarks/test_engine_benchmark.py -q -s` |
| **`engine.run()` runtime** | **15.37 s** (51 % of the 30 s budget) |
| **Normalized peak RSS** | **469,549,056 bytes ≈ 0.47 GB** (31 % of the 1.5 GB budget) |
| Orders / fills | 19 / 19 |

The whole pytest invocation reports about 20 s; the extra ~5 s is dataset
construction and interpreter startup, which the assertion deliberately excludes.

## Regression policy

A change is a regression when, re-measured on comparable hardware in its own
process, either number is **more than 20 % worse** than the reference above:

- runtime above **18.4 s**;
- normalized peak RSS above **563,458,867 bytes (≈ 0.56 GB)**.

A regression must be explained and either fixed or accepted deliberately, with
this table updated in the same commit. The hard budgets (30 s, 1.5 GB) are the
failure threshold, not the target: crossing 20 % while still passing the gate
is exactly the drift this policy exists to catch.

Both numbers are machine-specific. Re-measure before comparing, and record the
CPU, OS, Python, and NumPy versions alongside any new figure.
