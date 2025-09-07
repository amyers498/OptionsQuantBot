import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.scheduler.runner import run_premarket, run_intraday, run_eod
from src.utils.logging import get_logger
from src.utils.time import set_local_tz
from src.utils.config import load_config


logger = get_logger(__name__)


def _init_env() -> None:
    # Load .env if present
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def _setup_timezone(cfg: dict) -> None:
    tz_name = cfg.get("timezone", "America/New_York")
    set_local_tz(tz_name)


def main() -> None:
    _init_env()

    parser = argparse.ArgumentParser(description="Alpaca Options Quant — Paper Trader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("premarket", help="Run premarket workflow")
    sub.add_parser("intraday", help="Run intraday polling loop once")
    sub.add_parser("eod", help="Run end-of-day workflow")

    args = parser.parse_args()

    cfg = load_config(Path("config.yaml"))
    _setup_timezone(cfg)

    env = cfg.get("env", "paper").lower()
    if env != "paper":
        logger.warning("Non-paper env configured. Ensure live trading is disabled!")

    if args.cmd == "premarket":
        logger.info("Starting premarket workflow")
        run_premarket(cfg)
    elif args.cmd == "intraday":
        logger.info("Starting intraday workflow (single tick)")
        run_intraday(cfg)
    elif args.cmd == "eod":
        logger.info("Starting end-of-day workflow")
        run_eod(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

