from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
import datetime as dt


def parse_occ_strike(occ_symbol: str) -> Optional[float]:
    try:
        digits = occ_symbol[-8:]
        val = int(digits)
        return val / 1000.0
    except Exception:
        return None


def parse_occ_expiry(occ_symbol: str) -> Optional[dt.date]:
    """Parse OCC option symbol expiry date (YYMMDD) before the C/P flag.
    Example: AAPL250117C00190000 -> 2025-01-17
    """
    try:
        # Find last occurrence of 'C' or 'P'
        idx_c = occ_symbol.rfind('C')
        idx_p = occ_symbol.rfind('P')
        idx = max(idx_c, idx_p)
        if idx == -1:
            return None
        yymmdd = occ_symbol[idx-6:idx]
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        year = 2000 + yy
        return dt.date(year, mm, dd)
    except Exception:
        return None


def leg_value(mid: Optional[float], qty: int, side: Literal["LONG", "SHORT"]) -> float:
    if mid is None:
        return 0.0
    sign = 1 if side == "LONG" else -1
    return sign * (mid * 100.0 * qty)


def spread_max_value(width: float, qty: int) -> float:
    return max(0.0, width) * 100.0 * qty


def spread_current_value(legs_json: List[Dict[str, Any]], snapshots: Dict[str, Dict[str, Any]]) -> float:
    total = 0.0
    for leg in legs_json:
        occ = leg.get("symbol")
        qty = int(leg.get("qty", 1))
        side = str(leg.get("side", "buy")).upper()
        snap = snapshots.get(occ, {}) if occ else {}
        q = snap.get("latest_quote", {})
        bp = q.get("bid_price")
        ap = q.get("ask_price")
        if bp and ap and bp > 0 and ap > 0:
            mid = (bp + ap) / 2.0
            val = mid * 100.0 * qty
            total += val if side == "BUY" else -val
    return total


def legs_current_value(legs_json: List[Dict[str, Any]], snapshots: Dict[str, Dict[str, Any]]) -> float:
    """Generic current value for any set of option legs from snapshots.
    Long legs add, short legs subtract.
    """
    total = 0.0
    for leg in legs_json:
        occ = leg.get("symbol")
        qty = int(leg.get("qty", 1))
        side = str(leg.get("side", "buy")).upper()
        snap = snapshots.get(occ, {}) if occ else {}
        q = snap.get("latest_quote", {})
        bp = q.get("bid_price"); ap = q.get("ask_price")
        if bp and ap and bp > 0 and ap > 0:
            mid = (bp + ap) / 2.0
            val = mid * 100.0 * qty
            total += val if side == "BUY" else -val
    return total


def spread_width_from_legs(legs_json: List[Dict[str, Any]]) -> Optional[float]:
    strikes = []
    for leg in legs_json:
        occ = leg.get("symbol")
        k = parse_occ_strike(occ or "")
        if k is not None:
            strikes.append(k)
    if len(strikes) < 2:
        return None
    return abs(max(strikes) - min(strikes))
