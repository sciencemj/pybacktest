"""Resource-bounded regressions for Decimal exponent-limit behavior."""

import importlib.util
import json
import os
import subprocess
import sys
from decimal import MAX_EMAX, MIN_EMIN, Decimal
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TIMEOUT_SECONDS = 10
pytestmark = pytest.mark.skipif(
    os.name != "posix" or importlib.util.find_spec("resource") is None,
    reason="resource-limited subprocess tests require POSIX resource limits",
)

_BOUNDARY_SCRIPT = """
import json
import resource
import sys
from datetime import UTC, datetime
from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from zoneinfo import ZoneInfo

from pybacktest.domain.events import OrderRejected
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Order,
    OrderSide,
    OrderType,
    TargetWeight,
    TimeInForce,
)
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.ports.risk import RiskContext
from pybacktest.risk.policies import LongShortRisk
from pybacktest.risk.sizing import DefaultOrderSizer
from pybacktest.risk._decimal import floor_quantity_to_lot

# Darwin reserves roughly 415 GiB of virtual address space for the shared
# cache before Python imports project code; retain less than 1 GiB headroom.
MEMORY_LIMIT_BYTES = (
    416 * 1024 * 1024 * 1024
    if sys.platform == "darwin"
    else 512 * 1024 * 1024
)
CPU_LIMIT_SECONDS = 5
resource.setrlimit(
    resource.RLIMIT_AS,
    (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES),
)
resource.setrlimit(
    resource.RLIMIT_CPU,
    (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS),
)

NOW = datetime(2024, 1, 2, tzinfo=UTC)
ZERO_MONEY = Money.usd("0")
HUGE = Decimal(f"1E+{MAX_EMAX - 2}")
SMALL = Decimal(f"1E{MIN_EMIN}")


def context(*, cash, price, tick, lot):
    item = Instrument(
        id=InstrumentId(venue="XNAS", symbol="BOUNDARY"),
        quote_currency="USD",
        tick_size=tick,
        lot_size=lot,
        timezone=ZoneInfo("UTC"),
    )
    with localcontext() as setup:
        setup.prec = 128
        setup.Emax = MAX_EMAX
        setup.Emin = MIN_EMIN
        snapshot = PortfolioSnapshot(
            timestamp=NOW,
            cash=Money.usd(cash),
            positions={},
            realized_pnl=ZERO_MONEY,
            unrealized_pnl=ZERO_MONEY,
            total_fees=ZERO_MONEY,
            market_value=ZERO_MONEY,
            gross_exposure=ZERO_MONEY,
            equity=Money.usd(cash),
            valuation_prices={},
            cash_events=(),
        )
    return item, RiskContext(
        snapshot=snapshot,
        prices={item.id: Money.usd(price)},
        instruments={item.id: item},
        tradable=frozenset({item.id}),
        order_id=OrderId.parse("order_" + "a" * 32),
        submitted_at=NOW,
        active_from=NOW,
        active_orders=(),
    )


def order(item, *, side, quantity):
    return Order.pending(
        id=OrderId.parse("order_" + "b" * 32),
        instrument=item.id,
        side=side,
        type=OrderType.MARKET,
        quantity=Quantity.of(quantity),
        quote_currency="USD",
        limit_price=None,
        time_in_force=TimeInForce.DAY,
        submitted_at=NOW,
        active_from=NOW,
        reason=DecisionReason.of("boundary"),
    )


def risk_payload(decision):
    return {
        "kind": "risk",
        "status": decision.status.value,
        "final_quantity": str(decision.final_quantity.value),
        "codes": list(decision.codes),
    }


try:
    case = sys.argv[1]
    if case == "max_sizing":
        item, current = context(
            cash=HUGE,
            price=HUGE,
            tick=Decimal("1"),
            lot=Decimal("1"),
        )
        result = DefaultOrderSizer().size(
            TargetWeight(
                instrument=item.id,
                weight=Decimal("1"),
                reason=DecisionReason.of("max_sizing"),
            ),
            current,
        )
        if isinstance(result, OrderRejected):
            payload = {
                "kind": "sizing_rejection",
                "code": result.reason.code,
            }
        else:
            payload = {
                "kind": "order",
                "side": result.side.value,
                "quantity": str(result.quantity.value),
            }
    elif case == "max_risk":
        item, current = context(
            cash=HUGE,
            price=HUGE,
            tick=HUGE,
            lot=Decimal("1"),
        )
        payload = risk_payload(
            LongShortRisk(
                max_leverage=Decimal("100"),
                max_position_weight=Decimal("100"),
                allow_short=True,
            ).evaluate(
                order(item, side=OrderSide.SELL, quantity="101"),
                current,
            )
        )
    elif case == "min_risk":
        item, current = context(
            cash=SMALL,
            price=SMALL,
            tick=SMALL,
            lot=SMALL,
        )
        payload = risk_payload(
            LongShortRisk(
                max_leverage=Decimal("1"),
                max_position_weight=Decimal("1"),
                allow_short=True,
            ).evaluate(
                order(item, side=OrderSide.BUY, quantity=SMALL),
                current,
            )
        )
    elif case == "unrepresentable_risk":
        item, current = context(
            cash=HUGE,
            price=SMALL,
            tick=SMALL,
            lot=SMALL,
        )
        payload = risk_payload(
            LongShortRisk(
                max_leverage=Decimal("1"),
                max_position_weight=Decimal("1"),
                allow_short=True,
            ).evaluate(
                order(item, side=OrderSide.BUY, quantity=SMALL),
                current,
            )
        )
    elif case == "mixed_aligned_sizing":
        item, current = context(
            cash=HUGE,
            price=Decimal("1"),
            tick=Decimal("1"),
            lot=SMALL,
        )
        result = DefaultOrderSizer().size(
            TargetWeight(
                instrument=item.id,
                weight=Decimal("1"),
                reason=DecisionReason.of("mixed_aligned_sizing"),
            ),
            current,
        )
        if isinstance(result, OrderRejected):
            payload = {
                "kind": "sizing_rejection",
                "code": result.reason.code,
            }
        else:
            payload = {
                "kind": "order",
                "side": result.side.value,
                "quantity": str(result.quantity.value),
            }
    elif case == "mixed_aligned_risk":
        item, current = context(
            cash=HUGE,
            price=Decimal("1"),
            tick=Decimal("1"),
            lot=SMALL,
        )
        policy = LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=Decimal("1"),
            allow_short=True,
        )
        payload = {
            "kind": "mixed_risk",
            "buy": risk_payload(
                policy.evaluate(
                    order(item, side=OrderSide.BUY, quantity=HUGE),
                    current,
                )
            ),
            "sell": risk_payload(
                policy.evaluate(
                    order(item, side=OrderSide.SELL, quantity=HUGE),
                    current,
                )
            ),
        }
    elif case == "lowered_int_string_limit":
        sys.set_int_max_str_digits(640)
        notional = Decimal("3" + "0" * 698 + "1")
        expected = Decimal("1" + "0" * 699)
        direct = floor_quantity_to_lot(
            notional,
            Decimal("3"),
            Decimal("1"),
        )
        item, current = context(
            cash=notional,
            price=Decimal("3"),
            tick=Decimal("1"),
            lot=Decimal("1"),
        )
        sized = DefaultOrderSizer().size(
            TargetWeight(
                instrument=item.id,
                weight=Decimal("1"),
                reason=DecisionReason.of("lowered_int_string_limit"),
            ),
            current,
        )
        risk = LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=Decimal("1"),
            allow_short=True,
        ).evaluate(
            order(
                item,
                side=OrderSide.SELL,
                quantity=expected,
            ),
            current,
        )
        payload = {
            "kind": "int_limit",
            "direct": str(direct),
            "sizing": (
                {
                    "kind": "rejection",
                    "code": sized.reason.code,
                }
                if isinstance(sized, OrderRejected)
                else {
                    "kind": "order",
                    "side": sized.side.value,
                    "quantity": str(sized.quantity.value),
                }
            ),
            "risk": risk_payload(risk),
        }
    else:
        raise AssertionError(f"unknown boundary case: {case}")
except BaseException as error:
    payload = {
        "kind": "error",
        "error_type": type(error).__name__,
        "message": str(error),
    }

print(json.dumps(payload))
"""


def _run_boundary_case(case: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", _BOUNDARY_SCRIPT, case],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, (
        f"boundary subprocess exited {completed.returncode}: {completed.stderr}"
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    assert payload.get("kind") != "error", payload
    return payload


def test_max_emax_mark_alignment_sizes_one_exact_share_without_exhaustion():
    payload = _run_boundary_case("max_sizing")

    assert payload["kind"] == "order"
    assert payload["side"] == "buy"
    assert Decimal(str(payload["quantity"])) == Decimal("1")


def test_max_emax_loose_risk_caps_adjust_101_to_100_without_exhaustion():
    payload = _run_boundary_case("max_risk")

    assert payload["kind"] == "risk"
    assert payload["status"] == "adjusted"
    assert Decimal(str(payload["final_quantity"])) == Decimal("100")
    assert payload["codes"] == [
        "max_position_weight",
        "max_leverage",
    ]


def test_min_emin_lot_cancellation_passes_the_exact_small_order():
    payload = _run_boundary_case("min_risk")

    assert payload["kind"] == "risk"
    assert payload["status"] == "passed"
    assert Decimal(str(payload["final_quantity"])) == Decimal(f"1E{MIN_EMIN}")
    assert payload["codes"] == []


def test_inherently_unrepresentable_cap_is_a_typed_rejection():
    payload = _run_boundary_case("unrepresentable_risk")

    assert payload["kind"] == "risk"
    assert payload["status"] == "rejected"
    assert Decimal(str(payload["final_quantity"])) == Decimal("0")
    assert payload["codes"] == ["invalid_risk_arithmetic"]


def test_mixed_max_min_aligned_weight_sizing_uses_compact_raw_quantity():
    payload = _run_boundary_case("mixed_aligned_sizing")

    assert payload["kind"] == "order"
    assert payload["side"] == "buy"
    assert Decimal(str(payload["quantity"])) == Decimal(f"1E+{MAX_EMAX - 2}")


def test_mixed_max_min_aligned_loose_risk_caps_pass_both_directions():
    payload = _run_boundary_case("mixed_aligned_risk")

    assert payload["kind"] == "mixed_risk"
    for side in ("buy", "sell"):
        decision = payload[side]
        assert isinstance(decision, dict)
        assert decision["status"] == "passed"
        assert Decimal(str(decision["final_quantity"])) == Decimal(f"1E+{MAX_EMAX - 2}")
        assert decision["codes"] == []


def test_lowered_integer_string_limit_preserves_direct_and_public_results():
    payload = _run_boundary_case("lowered_int_string_limit")
    expected = Decimal("1" + "0" * 699)

    assert payload["kind"] == "int_limit"
    assert Decimal(str(payload["direct"])) == expected
    sizing = payload["sizing"]
    assert isinstance(sizing, dict)
    assert sizing["kind"] == "order"
    assert sizing["side"] == "buy"
    assert Decimal(str(sizing["quantity"])) == expected
    risk = payload["risk"]
    assert isinstance(risk, dict)
    assert risk["status"] == "passed"
    assert Decimal(str(risk["final_quantity"])) == expected
    assert risk["codes"] == []
