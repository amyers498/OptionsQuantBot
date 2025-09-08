from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
import datetime as dt
import math


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


def parse_occ_underlying(occ_symbol: str) -> Optional[str]:
    """Extract the OCC underlying root from a standard OCC symbol.
    Example: AAPL250117C00190000 -> AAPL
    """
    try:
        idx_c = occ_symbol.rfind('C')
        idx_p = occ_symbol.rfind('P')
        idx = max(idx_c, idx_p)
        if idx < 6:
            return None
        # Underlying is everything before the 6-digit YYMMDD preceding C/P
        return occ_symbol[: idx - 6]
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bsm_greeks(spot: float, strike: float, t_years: float, iv: float, r: float = 0.01, is_call: bool = True) -> Dict[str, float]:
    """Black-Scholes greeks using continuous compounding.

    Returns delta, theta (per day), vega (per 1% vol change), all per 1 contract of underlying.
    """
    try:
        S = max(1e-9, float(spot))
        K = max(1e-9, float(strike))
        T = max(1e-9, float(t_years))
        sigma = max(1e-9, float(iv))
        r = float(r)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)
        pdf_d1 = _norm_pdf(d1)
        if is_call:
            delta = Nd1
            theta_annual = -(S * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * math.exp(-r * T) * Nd2
        else:
            delta = Nd1 - 1.0
            theta_annual = -(S * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        # Convert annual theta to per-day
        theta_per_day = theta_annual / 365.0
        # Vega per 1% change in vol (divide by 100)
        vega = (S * pdf_d1 * sqrtT) / 100.0
        return {"delta": delta, "theta": theta_per_day, "vega": vega}
    except Exception:
        return {"delta": 0.0, "theta": 0.0, "vega": 0.0}


def _bs_price(spot: float, strike: float, t_years: float, iv: float, r: float, is_call: bool) -> float:
    try:
        S = max(1e-9, float(spot)); K = max(1e-9, float(strike)); T = max(1e-9, float(t_years)); sigma = max(1e-9, float(iv)); r = float(r)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1 = _norm_cdf(d1); Nd2 = _norm_cdf(d2)
        disc = math.exp(-r * T)
        if is_call:
            return S * Nd1 - K * disc * Nd2
        else:
            return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    except Exception:
        return 0.0


def implied_vol_bsm(spot: float, strike: float, t_years: float, price: float, r: float = 0.01, is_call: bool = True) -> Optional[float]:
    """Invert Black-Scholes to estimate IV from option mid price using bisection.
    Returns None if inversion fails.
    """
    try:
        S = float(spot); K = float(strike); T = float(t_years); P = float(price)
        if S <= 0 or K <= 0 or T <= 0 or P <= 0:
            return None
        lo, hi = 1e-4, 5.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            pm = _bs_price(S, K, T, mid, r, is_call)
            if abs(pm - P) < 1e-4:
                return mid
            if pm > P:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    except Exception:
        return None
