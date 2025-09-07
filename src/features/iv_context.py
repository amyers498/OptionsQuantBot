from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ATMContext:
    atm_iv: Optional[float]
    atm_call_mid: Optional[float]
    atm_put_mid: Optional[float]
    expiry: Optional[str]
    strike: Optional[float]


def compute_iv_percentile(values: List[float], current_iv: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    rank = sum(1 for v in sorted_vals if v <= current_iv)
    return 100.0 * rank / len(sorted_vals)

