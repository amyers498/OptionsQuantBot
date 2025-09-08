from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class OptionType(enum.Enum):
    C = "C"
    P = "P"


class PositionStructure(enum.Enum):
    CALL = "CALL"
    PUT = "PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"


class PositionStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class TradeSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(16), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    close = Column(Float, nullable=False)
    sma20 = Column(Float, nullable=True)
    sma50 = Column(Float, nullable=True)
    rsi14 = Column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_daily_prices_ticker_date"),)


class OptionSnapshot(Base):
    __tablename__ = "option_snapshots"

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, index=True, nullable=False)
    ticker = Column(String(16), index=True, nullable=False)
    expiry = Column(Date, index=True, nullable=False)
    strike = Column(Float, nullable=False)
    type = Column(Enum(OptionType), nullable=False)

    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    mid = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    iv = Column(Float, nullable=True)
    oi = Column(Integer, nullable=True)
    volume = Column(Integer, nullable=True)


class IVSummary(Base):
    __tablename__ = "iv_summary"

    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True, nullable=False)
    ticker = Column(String(16), index=True, nullable=False)
    atm_strike = Column(Float, nullable=True)
    expiry = Column(Date, nullable=True)
    atm_iv = Column(Float, nullable=True)
    iv_percentile_lookback = Column(Integer, nullable=True)
    atm_iv_percentile = Column(Float, nullable=True)  # 0–100

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_iv_summary_ticker_date"),)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(16), index=True, nullable=False)
    structure = Column(Enum(PositionStructure), nullable=False)
    legs_json = Column(JSON, nullable=False)
    opened_at = Column(DateTime, index=True, nullable=False)
    status = Column(Enum(PositionStatus), index=True, nullable=False, default=PositionStatus.OPEN)
    entry_debit = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    direction = Column(String(8), nullable=True)
    closed_value = Column(Float, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    targets_json = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)

    trades = relationship("Trade", back_populates="position")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id", ondelete="CASCADE"), index=True, nullable=False)
    side = Column(Enum(TradeSide), nullable=False)
    leg_ref = Column(Integer, nullable=True)
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    ts = Column(DateTime, index=True, nullable=False)
    order_id = Column(String(64), index=True, nullable=True)
    slippage = Column(Float, nullable=True)

    position = relationship("Position", back_populates="trades")


class MetricsDaily(Base):
    __tablename__ = "metrics_daily"

    date = Column(Date, primary_key=True)
    cash = Column(Float, nullable=True)
    equity = Column(Float, nullable=True)
    realized_pl_day = Column(Float, nullable=True)
    unrealized_pl = Column(Float, nullable=True)
    total_pl = Column(Float, nullable=True)
    drawdown_from_peak = Column(Float, nullable=True)
    net_delta = Column(Float, nullable=True)
    net_theta = Column(Float, nullable=True)
    net_vega = Column(Float, nullable=True)
    open_risk = Column(Float, nullable=True)


class OrdersOutbox(Base):
    __tablename__ = "orders_outbox"

    id = Column(Integer, primary_key=True)
    key = Column(String(128), unique=True, index=True, nullable=False)
    payload_json = Column(JSON, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    status = Column(String(32), nullable=False, default="PENDING")
