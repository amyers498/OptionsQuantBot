from __future__ import annotations

from typing import Optional


def implied_move_abs(atm_call_mid: Optional[float], atm_put_mid: Optional[float]) -> Optional[float]:
    if atm_call_mid is None or atm_put_mid is None:
        return None
    return atm_call_mid + atm_put_mid

