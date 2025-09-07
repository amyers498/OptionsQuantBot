from __future__ import annotations

from typing import List, Literal, Optional


def leg_value(mid: Optional[float], qty: int, side: Literal["LONG", "SHORT"]) -> float:
    if mid is None:
        return 0.0
    sign = 1 if side == "LONG" else -1
    return sign * (mid * 100.0 * qty)


def spread_max_value(width: float, qty: int) -> float:
    return max(0.0, width) * 100.0 * qty

