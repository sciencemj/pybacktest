"""Execute the README so its documentation cannot drift from the code.

Every fenced Python block in ``README.md`` must be classified by its fence
info string, and every classification is checked here:

``` ```python ```               executed in a fresh namespace
``` ```python title="not-executed: <reason>" ```
                                compiled only, and the reason must be present
``` ```python title="requires: <modules>" ```
                                compiled always, executed when the optional
                                dependency is installed

An unrecognized info string fails, so no block can silently opt out of being
checked.
"""

import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"

CONSTRUCTOR_IO_STATEMENT = (
    "Constructors perform no network or filesystem I/O; data loading begins "
    "at `reset()` / `run()`."
)
OPTIONAL_ADAPTER_IMPORT = "from pybacktest.adapters.data import ParquetDataSource"

_FENCE_DELIMITER = re.compile(
    r"^ {0,3}(?P<delimiter>`{3,}|~{3,})(?P<info>[^\r\n]*)(?:\r?\n)?$"
)
_NOT_EXECUTED = re.compile(r'^title="not-executed:\s*(?P<reason>.+?)"$')
_REQUIRES = re.compile(r'^title="requires:\s*(?P<modules>[A-Za-z0-9_,\s]+)"$')


@dataclass(frozen=True, slots=True)
class _Block:
    index: int
    info: str
    source: str
    reason: str | None
    requires: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"README.md:block{self.index}"

    @property
    def is_core(self) -> bool:
        return self.reason is None and not self.requires


def _classify(index: int, info: str, source: str) -> _Block:
    stripped = info.strip()
    if not stripped:
        return _Block(index, stripped, source, None, ())
    not_executed = _NOT_EXECUTED.match(stripped)
    if not_executed is not None:
        return _Block(
            index,
            stripped,
            source,
            not_executed.group("reason").strip(),
            (),
        )
    requires = _REQUIRES.match(stripped)
    if requires is not None:
        modules = tuple(
            module.strip()
            for module in requires.group("modules").split(",")
            if module.strip()
        )
        return _Block(index, stripped, source, None, modules)
    raise AssertionError(
        f"README.md block {index} has an unrecognized fence info string: "
        f"{stripped!r}. Use no tag, 'not-executed: <reason>', or "
        "'requires: <modules>'."
    )


def _is_closing_fence(line: str, delimiter: str) -> bool:
    marker = re.escape(delimiter[0])
    minimum_length = len(delimiter)
    return (
        re.fullmatch(
            rf" {{0,3}}{marker}{{{minimum_length},}}[ \t]*(?:\r?\n)?",
            line,
        )
        is not None
    )


def _parse_blocks(text: str) -> tuple[_Block, ...]:
    blocks: list[_Block] = []
    delimiter: str | None = None
    python_info: str | None = None
    source_lines: list[str] = []
    opening_line = 0

    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        match = _FENCE_DELIMITER.match(line)
        if delimiter is None:
            if match is None:
                continue
            delimiter = match.group("delimiter")
            info = match.group("info")
            stripped_info = info.strip()
            if stripped_info == "python" or stripped_info.startswith("python "):
                python_info = stripped_info.removeprefix("python")
                source_lines = []
                opening_line = line_number
            continue

        if _is_closing_fence(line, delimiter):
            if python_info is not None:
                blocks.append(
                    _classify(
                        len(blocks),
                        python_info,
                        "".join(source_lines),
                    )
                )
            delimiter = None
            python_info = None
            source_lines = []
            opening_line = 0
            continue

        if (
            python_info is not None
            and match is not None
            and match.group("delimiter")[0] == delimiter[0]
        ):
            raise AssertionError(
                "README.md has a malformed closing Python fence "
                f"at line {line_number}; its opening fence is at line "
                f"{opening_line}."
            )
        if python_info is not None:
            source_lines.append(line)

    if delimiter is not None and python_info is not None:
        raise AssertionError(
            f"README.md has an unclosed Python fence at line {opening_line}."
        )
    return tuple(blocks)


def _blocks() -> tuple[_Block, ...]:
    return _parse_blocks(README.read_text(encoding="utf-8"))


BLOCKS = _blocks()


def _identifier(block: _Block) -> str:
    if block.reason is not None:
        return f"block{block.index}-not-executed"
    if block.requires:
        return f"block{block.index}-requires-{'-'.join(block.requires)}"
    return f"block{block.index}-core"


@pytest.mark.parametrize("block", BLOCKS, ids=_identifier)
def test_every_readme_block_compiles(block: _Block) -> None:
    compile(block.source, block.name, "exec")


@pytest.mark.parametrize(
    "block",
    [block for block in BLOCKS if block.reason is not None],
    ids=_identifier,
)
def test_non_executable_blocks_state_a_reason(block: _Block) -> None:
    assert block.reason


@pytest.mark.parametrize(
    "block",
    [block for block in BLOCKS if block.is_core],
    ids=_identifier,
)
def test_core_blocks_run_in_a_fresh_namespace(block: _Block) -> None:
    # Executing the documentation is the point of this test.
    exec(compile(block.source, block.name, "exec"), {"__name__": "__readme__"})


@pytest.mark.parametrize(
    "block",
    [block for block in BLOCKS if block.requires],
    ids=_identifier,
)
def test_optional_extra_blocks_run_when_their_dependency_exists(
    block: _Block,
) -> None:
    missing = [
        module for module in block.requires if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(f"{block.name} needs {', '.join(missing)}")
    # Executing the documentation is the point of this test.
    exec(compile(block.source, block.name, "exec"), {"__name__": "__readme__"})


def test_at_least_one_core_block_is_executed() -> None:
    assert [block for block in BLOCKS if block.is_core]


def test_no_block_is_silently_skipped() -> None:
    assert BLOCKS
    for block in BLOCKS:
        assert block.is_core or block.reason is not None or block.requires


def test_unclosed_python_fence_is_rejected(tmp_path: Path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("```python\nvalue = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "README", readme)

    with pytest.raises(AssertionError, match="Python fence"):
        _blocks()


def test_malformed_python_closing_fence_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "```python\nvalue = 1\n``` trailing text\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "README", readme)

    with pytest.raises(AssertionError, match="Python fence"):
        _blocks()


def test_parquet_example_runs_a_typed_backtest_with_a_real_fill() -> None:
    parquet_blocks = [
        block for block in BLOCKS if block.requires == ("pandas", "pyarrow")
    ]
    assert len(parquet_blocks) == 1
    missing = [
        module
        for module in parquet_blocks[0].requires
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(f"Parquet example needs {', '.join(missing)}")

    namespace = {"__name__": "__readme_parquet__"}
    block = parquet_blocks[0]
    exec(compile(block.source, block.name, "exec"), namespace)

    from pybacktest import (
        BacktestEngine,
        BacktestRequest,
        BacktestResult,
        LongShortRisk,
        Money,
        Quantity,
        SimulatedBrokerFactory,
    )

    assert isinstance(namespace.get("broker_factory"), SimulatedBrokerFactory)
    assert isinstance(namespace.get("risk_policy"), LongShortRisk)
    assert isinstance(namespace.get("engine"), BacktestEngine)
    assert isinstance(namespace.get("request"), BacktestRequest)
    result = namespace.get("result")
    assert isinstance(result, BacktestResult)
    assert len(result.orders) == 1
    assert result.fills[0].price == Money.usd("104.00")
    assert result.fills[0].quantity == Quantity.of("49")


def test_core_examples_run_when_pyarrow_is_not_installed() -> None:
    probe = """
import builtins
import importlib

from tests.integration.test_readme import BLOCKS

real_import = builtins.__import__
real_import_module = importlib.import_module


def import_statement_without_pyarrow(
    name, globals=None, locals=None, fromlist=(), level=0
):
    if name == "pyarrow" or name.startswith("pyarrow."):
        raise ModuleNotFoundError("pyarrow is intentionally unavailable")
    return real_import(name, globals, locals, fromlist, level)


def import_without_pyarrow(name, package=None):
    if name == "pyarrow" or name.startswith("pyarrow."):
        raise ModuleNotFoundError("pyarrow is intentionally unavailable")
    return real_import_module(name, package)


builtins.__import__ = import_statement_without_pyarrow
importlib.import_module = import_without_pyarrow
for block in BLOCKS:
    if block.is_core:
        namespace = {"__name__": "__readme_core_probe__"}
        exec(compile(block.source, block.name, "exec"), namespace)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_readme_states_the_constructor_io_boundary_exactly() -> None:
    assert CONSTRUCTOR_IO_STATEMENT in README.read_text(encoding="utf-8")


def test_readme_documents_the_explicit_optional_adapter_import() -> None:
    assert OPTIONAL_ADAPTER_IMPORT in README.read_text(encoding="utf-8")
