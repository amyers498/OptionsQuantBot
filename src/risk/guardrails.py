from __future__ import annotations


def within_daily_loss_limit(realized_today: float, equity_start: float, loss_limit_pct: float) -> bool:
    if equity_start <= 0:
        return False
    return realized_today >= -abs(loss_limit_pct) / 100.0 * equity_start


def within_open_risk(open_risk: float, equity: float, max_open_risk_pct: float) -> bool:
    if equity <= 0:
        return False
    return open_risk <= (max_open_risk_pct / 100.0) * equity

