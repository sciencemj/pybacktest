"""Stable exceptions raised by the Pybacktest domain layer."""


class PybacktestError(Exception):
    """Base class for stable Pybacktest failures."""


class ConfigurationError(PybacktestError, ValueError):
    """Raised when typed configuration violates a declared invariant."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class DataValidationError(PybacktestError, ValueError):
    """Raised when market data violates the source contract."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class LookaheadViolation(PybacktestError):
    """Raised when code requests data after the current engine timestamp."""


class AccountingInvariantError(PybacktestError):
    """Raised when a ledger transition cannot be reconciled."""


class ClockRegressionError(PybacktestError):
    """Raised when simulation time moves backwards."""


class AdapterContractError(PybacktestError):
    """Raised when a port implementation violates its contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "adapter_contract_error",
    ) -> None:
        super().__init__(message)
        self.code = code
