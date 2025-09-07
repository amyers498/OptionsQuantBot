# Alpaca Options Quant (No-News) — Paper Trading Bot

A rule-based options paper-trader for Alpaca, focused on price-trend and IV-regime signals with liquidity guardrails, risk controls, and Discord notifications.

This repository contains a clean, modular implementation scaffold ready to be wired to Alpaca's Market Data and Paper Trading APIs.

## Quick Start

- Python: 3.11+
- Timezone: America/New_York

1) Copy `config.yaml` and `.env.example` to `.env` with your keys.
2) Create a virtualenv and install dependencies listed in `pyproject.toml`.
3) Run CLI commands:

- `python -m src.app premarket`
- `python -m src.app intraday`
- `python -m src.app eod`

## Structure

See `src/` for modules:

- `alpaca_client/` API clients (auth, market data, options, orders, account)
- `data/` SQLite engine, ORM models, repositories
- `features/` Trend, IV context, liquidity, implied move
- `selector/` Strategy mapping and strike/expiry picker
- `risk/` Sizing, guardrails
- `pnl/` Valuation, metrics, rollups
- `notify/` Discord webhook sender
- `scheduler/` Basic orchestration
- `utils/` Logging, time, idempotency

## Important Notes

- Paper trading only; add an explicit config gate before enabling live.
- Use mid prices for valuation and slippage calculations.
- Keep all time math in America/New_York.

## Testing

- Unit tests in `tests/unit/` for indicators, liquidity filters, mapping, and P&L math.
- Integration tests in `tests/integration/` with Alpaca client mocks and Discord webhook mock.

## Licensing

No license specified. Do not redistribute without permission.
