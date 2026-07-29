"""Stable exceptions raised by the Pybacktest domain layer."""


class PybacktestError(Exception):
    """Base class for stable Pybacktest failures."""


class ConfigurationError(PybacktestError, ValueError):
    """Raised when typed configuration violates a declared invariant."""


class DataValidationError(PybacktestError, ValueError):
    """Raised when market data violates the source contract."""


class LookaheadViolation(PybacktestError):
    """Raised when code requests data after the current engine timestamp."""


class AccountingInvariantError(PybacktestError):
    """Raised when a ledger transition cannot be reconciled."""


class ClockRegressionError(PybacktestError):
    """Raised when simulation time moves backwards."""


class AdapterContractError(PybacktestError):
    """Raised when a port implementation violates its contract."""
