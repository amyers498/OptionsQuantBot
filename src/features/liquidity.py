from __future__ import annotations

from typing import Optional


def passes_liquidity(oi: Optional[int], vol: Optional[int], bid: Optional[float], ask: Optional[float], max_spread_pct: float) -> bool:
    if (oi or 0) < 1 or (vol or 0) < 1:
        return False
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return False
    mid = (bid + ask) / 2
    if mid <= 0:
        return False
    spread_pct = abs(ask - bid) / mid
    return spread_pct <= max_spread_pct

