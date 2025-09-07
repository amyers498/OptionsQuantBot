from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Contract:
    symbol: str
    expiry: str
    strike: float
    type: str  # C/P
    delta: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]


def pick_single_delta_band(contracts: List[Contract], lo: float, hi: float) -> Optional[Contract]:
    candidates = [c for c in contracts if c.delta is not None and lo <= abs(c.delta) <= hi]
    # Prefer nearest to 0.5 within band
    candidates.sort(key=lambda c: abs(abs(c.delta or 0) - 0.5))
    return candidates[0] if candidates else None

