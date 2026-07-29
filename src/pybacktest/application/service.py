"""Thin application service for Python strategy execution."""

from pybacktest.application.requests import BacktestRequest
from pybacktest.domain.errors import ConfigurationError
from pybacktest.engine.engine import BacktestEngine
from pybacktest.results.models import BacktestResult


class BacktestService:
    """Delegate Python execution directly to the deterministic engine."""

    def __init__(self, engine: BacktestEngine) -> None:
        if not isinstance(engine, BacktestEngine):
            raise ConfigurationError("engine must be a BacktestEngine.")
        self._engine = engine

    def run_python(self, request: BacktestRequest) -> BacktestResult:
        """Execute a Python strategy without adding another run path."""
        return self._engine.run(request)


__all__ = ["BacktestService"]
