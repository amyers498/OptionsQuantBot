from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioSnapshot:
    cash: float
    equity: float
    realized_pl_day: float
    unrealized_pl: float
    total_pl: float
    drawdown_from_peak: float | None = None

