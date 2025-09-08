from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.data.models import Base


DB_PATH = Path("bot.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def _migrate_sqlite() -> None:
    """Lightweight, idempotent migrations for SQLite.
    Adds missing columns introduced after initial DB creation.
    """
    with ENGINE.begin() as conn:
        # Check existing columns on positions
        cols = set()
        try:
            res = conn.exec_driver_sql("PRAGMA table_info(positions)")
            cols = {row[1] for row in res}
        except Exception:
            cols = set()
        add_stmts = []
        if "direction" not in cols:
            add_stmts.append("ALTER TABLE positions ADD COLUMN direction VARCHAR(8)")
        if "closed_value" not in cols:
            add_stmts.append("ALTER TABLE positions ADD COLUMN closed_value REAL")
        if "closed_at" not in cols:
            add_stmts.append("ALTER TABLE positions ADD COLUMN closed_at DATETIME")
        for stmt in add_stmts:
            try:
                conn.exec_driver_sql(stmt)
            except Exception:
                pass


def init_db() -> None:
    Base.metadata.create_all(ENGINE)
    # Best-effort migrations for SQLite
    try:
        _migrate_sqlite()
    except Exception:
        pass


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
