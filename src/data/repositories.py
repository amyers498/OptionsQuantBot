from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.models import DailyPrice, OptionSnapshot, IVSummary, Position, Trade, MetricsDaily, OrdersOutbox


class PricesRepo:
    def __init__(self, session: Session):
        self.s = session

    def upsert_daily(self, rows: Iterable[DailyPrice]) -> None:
        for r in rows:
            existing = self.s.execute(
                select(DailyPrice).where(DailyPrice.ticker == r.ticker, DailyPrice.date == r.date)
            ).scalar_one_or_none()
            if existing:
                existing.close = r.close
                existing.sma20 = r.sma20
                existing.sma50 = r.sma50
                existing.rsi14 = r.rsi14
            else:
                self.s.add(r)


class OptionsRepo:
    def __init__(self, session: Session):
        self.s = session

    def add_snapshots(self, rows: Iterable[OptionSnapshot]) -> None:
        for r in rows:
            self.s.add(r)

    def upsert_iv_summary(self, row: IVSummary) -> None:
        existing = self.s.execute(
            select(IVSummary).where(IVSummary.ticker == row.ticker, IVSummary.date == row.date)
        ).scalar_one_or_none()
        if existing:
            existing.atm_strike = row.atm_strike
            existing.expiry = row.expiry
            existing.atm_iv = row.atm_iv
            existing.iv_percentile_lookback = row.iv_percentile_lookback
            existing.atm_iv_percentile = row.atm_iv_percentile
        else:
            self.s.add(row)


class PositionsRepo:
    def __init__(self, session: Session):
        self.s = session

    def add_position(self, p: Position) -> Position:
        self.s.add(p)
        self.s.flush()
        return p

    def add_trade(self, t: Trade) -> Trade:
        self.s.add(t)
        self.s.flush()
        return t


class MetricsRepo:
    def __init__(self, session: Session):
        self.s = session

    def upsert_daily(self, m: MetricsDaily) -> None:
        existing = self.s.get(MetricsDaily, m.date)
        if existing:
            for field in (
                "cash",
                "equity",
                "realized_pl_day",
                "unrealized_pl",
                "total_pl",
                "drawdown_from_peak",
                "net_delta",
                "net_theta",
                "net_vega",
                "open_risk",
            ):
                setattr(existing, field, getattr(m, field))
        else:
            self.s.add(m)


class OutboxRepo:
    def __init__(self, session: Session):
        self.s = session

    def put(self, key: str, payload_json: dict, status: str = "PENDING") -> OrdersOutbox:
        row = OrdersOutbox(key=key, payload_json=payload_json, status=status)
        self.s.add(row)
        self.s.flush()
        return row

    def get(self, key: str) -> Optional[OrdersOutbox]:
        return self.s.execute(select(OrdersOutbox).where(OrdersOutbox.key == key)).scalar_one_or_none()

