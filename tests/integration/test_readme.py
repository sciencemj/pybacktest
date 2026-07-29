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
from dataclasses import dataclass
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"

CONSTRUCTOR_IO_STATEMENT = (
    "Constructors perform no network or filesystem I/O; data loading begins "
    "at `reset()` / `run()`."
)
OPTIONAL_ADAPTER_IMPORT = "from pybacktest.adapters.data import ParquetDataSource"

_FENCE = re.compile(
    r"^```python(?P<info>[^\n]*)\n(?P<source>.*?)^```$",
    re.DOTALL | re.MULTILINE,
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


def _blocks() -> tuple[_Block, ...]:
    text = README.read_text(encoding="utf-8")
    return tuple(
        _classify(index, match.group("info"), match.group("source"))
        for index, match in enumerate(_FENCE.finditer(text))
    )


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


def test_readme_states_the_constructor_io_boundary_exactly() -> None:
    assert CONSTRUCTOR_IO_STATEMENT in README.read_text(encoding="utf-8")


def test_readme_documents_the_explicit_optional_adapter_import() -> None:
    assert OPTIONAL_ADAPTER_IMPORT in README.read_text(encoding="utf-8")
