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
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("run", help="Run long-lived scheduler (auto phases)")
    sub.add_parser("preflight", help="Run self-test probes and exit")
    sub.add_parser("premarket", help="Run premarket workflow")
    sub.add_parser("intraday", help="Run intraday polling loop once")
    sub.add_parser("eod", help="Run end-of-day workflow")

    args = parser.parse_args()

    cfg = load_config(Path("config.yaml"))
    _setup_timezone(cfg)

    env = cfg.get("env", "paper").lower()
    if env != "paper":
        logger.warning("Non-paper env configured. Ensure live trading is disabled!")

    # Default to 'run' if no subcommand provided
    cmd = args.cmd or "run"

    if cmd == "run":
        logger.info("Starting daemon scheduler (auto phases)")
        from src.scheduler.runner import run_daemon, run_self_test

        # Announce and probe before entering the loop
        try:
            run_self_test(cfg)
        except Exception as e:
            logger.error({"event": "self_test_error", "err": str(e)})
        run_daemon(cfg)
    elif cmd == "preflight":
        from src.scheduler.runner import run_self_test

        logger.info("Running self-test and exiting")
        run_self_test(cfg)
    elif cmd == "premarket":
        logger.info("Starting premarket workflow")
        run_premarket(cfg)
    elif cmd == "intraday":
        logger.info("Starting intraday workflow (single tick)")
        run_intraday(cfg)
    elif cmd == "eod":
        logger.info("Starting end-of-day workflow")
        run_eod(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
