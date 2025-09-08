from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from datetime import date

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.data.models import IVSummary


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


def upsert_iv_summary(
    session: Session,
    ticker: str,
    d: date,
    atm_strike: Optional[float],
    expiry: Optional[date],
    atm_iv: Optional[float],
    lookback_days: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Store today's ATM IV and compute percentile against recent history.

    Returns (atm_iv, percentile) where percentile is 0-100 or None if insufficient history.
    """
    if atm_iv is None:
        # store row with null iv; percentile remains None
        row = IVSummary(
            date=d,
            ticker=ticker,
            atm_strike=atm_strike,
            expiry=expiry,
            atm_iv=None,
            iv_percentile_lookback=lookback_days,
            atm_iv_percentile=None,
        )
        # upsert
        existing = session.execute(select(IVSummary).where(IVSummary.ticker == ticker, IVSummary.date == d)).scalar_one_or_none()
        if existing:
            existing.atm_strike = atm_strike
            existing.expiry = expiry
            existing.atm_iv = None
            existing.iv_percentile_lookback = lookback_days
            existing.atm_iv_percentile = None
        else:
            session.add(row)
        return None, None

    # Fetch last N days (excluding today)
    hist_q = (
        select(IVSummary.atm_iv)
        .where(IVSummary.ticker == ticker, IVSummary.date < d, IVSummary.atm_iv.isnot(None))
        .order_by(IVSummary.date.desc())
        .limit(lookback_days)
    )
    hist = [r[0] for r in session.execute(hist_q).all()]
    pct = compute_iv_percentile(hist, atm_iv) if hist else None

    existing = session.execute(select(IVSummary).where(IVSummary.ticker == ticker, IVSummary.date == d)).scalar_one_or_none()
    if existing:
        existing.atm_strike = atm_strike
        existing.expiry = expiry
        existing.atm_iv = atm_iv
        existing.iv_percentile_lookback = lookback_days
        existing.atm_iv_percentile = pct
    else:
        session.add(
            IVSummary(
                date=d,
                ticker=ticker,
                atm_strike=atm_strike,
                expiry=expiry,
                atm_iv=atm_iv,
                iv_percentile_lookback=lookback_days,
                atm_iv_percentile=pct,
            )
        )
    return atm_iv, pct
