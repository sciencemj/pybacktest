# Pybacktest 0.2

Pybacktest 0.2 is a deterministic, extensible Python backtesting engine.
Its approved design is documented in the
[Pybacktest V2 architecture design](docs/superpowers/specs/2026-07-29-pybacktest-v2-architecture-design.md).

API documentation will be filled in by Task 11.

## Explicit simulated broker configuration

Every execution assumption is selected with a typed model. A factory creates a
fresh broker for each run; no model is selected by a string or hidden default.

```python
from decimal import Decimal

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    PerShareCommission,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
    VolumeShareSlippage,
)

broker_factory = SimulatedBrokerFactory(
    fill_model=NextBarOpenFill(
        intrabar_policy=IntrabarPolicy.CONSERVATIVE,
    ),
    commission=PerShareCommission(
        rate_per_share=Decimal("0.005"),
    ),
    slippage=VolumeShareSlippage(
        impact_bps=Decimal("5"),
    ),
    liquidity=VolumeParticipationLimit(
        max_volume_ratio=Decimal("0.05"),
    ),
    borrow_cost=NoBorrowCost(),
)
```
