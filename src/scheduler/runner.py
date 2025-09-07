from __future__ import annotations

from typing import Dict, Any

from src.utils.logging import get_logger
from src.data.db import init_db


logger = get_logger(__name__)


def run_premarket(cfg: Dict[str, Any]) -> None:
    init_db()
    # TODO: Universe build, refresh daily prices, compute SMA, IV percentile, select strategies, stage orders
    logger.info({"event": "premarket_placeholder", "status": "ok"})


def run_intraday(cfg: Dict[str, Any]) -> None:
    # TODO: Update quotes/marks, evaluate exits/time stops, submit exits, send alerts
    logger.info({"event": "intraday_placeholder", "status": "ok"})


def run_eod(cfg: Dict[str, Any]) -> None:
    # TODO: Final MTM, compute account KPIs, roll IV percentile, update equity curve, send EOD report
    logger.info({"event": "eod_placeholder", "status": "ok"})

