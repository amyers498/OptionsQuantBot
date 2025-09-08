# Alpaca Options Quant (No-News) - Paper Trading Bot

A rule-based options paper-trader for Alpaca that uses price trend (SMA20/50), IV regime, and liquidity guardrails to select single calls/puts or defined-risk spreads. It runs on a long-lived scheduler, posts Discord alerts, and enforces budget and risk limits. Paper trading only.

## Requirements

- Python 3.11+
- Timezone: America/New_York
- Alpaca Paper Trading enabled with Options trading
- Options Market Data feed (indicative or OPRA) for snapshots

Dependencies (subset): httpx, pydantic, pandas, numpy, PyYAML, SQLAlchemy, tenacity, pytz, python-dotenv, websockets, msgpack

## Install

- Create venv and install:
  - Windows PowerShell:
    - `python -m venv .venv`
    - `.\\.venv\\Scripts\\Activate.ps1`
    - `pip install -e .`
- Copy `.env.example` to `.env` and fill keys:
  - `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`
  - `DISCORD_TRADES`, `DISCORD_EOD`

## Configuration

Edit `config.yaml` (ENV:VAR values resolve from `.env`):

- `alpaca.base_url_paper`: `https://paper-api.alpaca.markets`
- `alpaca.data_url`: `https://data.alpaca.markets`
- `alpaca.market_data_subscription`: equities feed (`iex` or `sip`)
- `alpaca.options_feed`: options feed (`indicative` or `opra`)
- `alpaca.contracts_source`: `auto` | `trading` | `data` (auto prefers trading_v2 for contracts discovery when data API is unavailable)
- `universe.top_n`, `universe.exclude`, `universe.static` (fallback list)
- `signals`: SMA windows, IV thresholds
- `liquidity`: min OI/volume, max spread pct
- `strategy`: expiry window, delta band, spread width
- `risk`: per-trade budget, max positions, open-risk cap, daily loss limit
- `exits`: time stop, targets, hard stop (scaffolded)
- `discord`: webhooks and colors
- `scheduling`: premarket cron, intraday cadence, EOD cron

## Run Modes

- Preflight (readiness check, exits):
  - `python -m src.app preflight`
- Daemon (auto phases by ET):
  - `python -m src.app`
- One-off phases:
  - `python -m src.app premarket`
  - `python -m src.app intraday`
  - `python -m src.app eod`

## How It Works

- Startup self-test posts to Discord:
  - Account snapshot, a Most-Actives sample, Options Probe (contracts endpoints + status), Options Stream auth result, Feeds in use.
  - Weekend Idle notice on Sat/Sun.
- Premarket (08:00 ET by default):
  - Universe build, trend signals (SMA20/50), options probe guard.
  - Announces Premarket Start (symbols) and a Risk Snapshot (cash, equity, buying power, open risk vs cap, positions).
  - For each symbol: discover contracts (prefers trading_v2), fetch snapshots (feed=indicative), map strategy, check liquidity and budgets, submit paper orders at mid; post [NEW].
  - Posts Premarket Summary (bull/bear/neutral counts and examples).
- Intraday (every N seconds):
  - Automated exits for singles (CALL/PUT): time stop, +50%/+100% targets, −50% hard stop, optional signal flip; submits sell_to_close and posts [EXIT] then [CLOSED] on fill.
  - Automated exits for spreads (BULL CALL/BEAR PUT): PT at % of max (default 70%), −50% hard stop, ≥5d time stop, optional signal flip; closes legs; posts [EXIT]/[CLOSED].
- EOD (16:15 ET):
  - Full synopsis: Cash, Equity, Realized (day), Unrealized, Total P/L, Open Risk, portfolio Greeks (Δ/Θ/ν), positions preview, and attribution by bucket.
  - Note: NTAs on paper sync next day.

## Strategy (Plain English)

- Direction: SMA20 vs SMA50; Bull if 20>50, Bear if 20<50, Neutral skip.
- IV regime: ATM IV of nearest 7–21 DTE; stores daily values and computes ticker‑specific percentile over ~90 days. Low ≤ 30th pctile; High ≥ 70th; otherwise Mid (if no history).
- Mapping:
  - Bull + Low IV -> Buy Call (delta ~ 0.35-0.55)
  - Bull + High/Mid IV -> Bull Call Spread (ATM long; short near implied-move or default width)
  - Bear + Low IV -> Buy Put
  - Bear + High/Mid IV -> Bear Put Spread
  - Neutral -> Skip
- Liquidity: require OI and day volume, spread% <= cap.
- Strikes/Expiry: singles pick delta-band; spreads long ATM, short near implied move or fixed width; ensure net debit within per-trade budget.

## Strategy (Playground Terms)

- If SMA20 > SMA50 then Direction=BULL; else if SMA20 < SMA50 then Direction=BEAR; else SKIP.
- Compute ATM IV on nearest expiry; IV_REGIME from 90‑day percentile (LOW if ≤30, HIGH if ≥70, else MID).
- If BULL and LOW -> CALL in 0.35<=|delta|<=0.55; if BULL and HIGH/MID -> BULL_CALL_SPREAD.
- If BEAR and LOW -> PUT in 0.35<=|delta|<=0.55; if BEAR and HIGH/MID -> BEAR_PUT_SPREAD.
- Passes liquidity if OI>=min, Vol>=min, (ask-bid)/mid <= max_spread_pct.
- For spreads: long=ATM; short strike=spot +/- implied_move_abs or default width; enforce per-trade budget.
- Submit paper limit orders at mid. Send Discord [NEW].

## Risk Controls

- Per‑trade budget: enforces `risk.per_trade_dollar_min` and `per_trade_dollar_max`.
- Max concurrent positions: enforces `risk.max_concurrent_positions` against broker positions.
- Funds check: skips candidates if net debit > available cash or > buying power.
- Min premium filter (`risk.min_premium`) + tighten spread cap to ~5% when premium < $0.80; skip outright < $0.40.
- Strict open‑risk cap: Σ(entry_debit of OPEN) + new_debit ≤ equity × `risk.max_open_risk_pct_of_equity`.
- Daily loss stop: config present; can be enabled on entries next.
- All skips are logged; no crashes on insufficient funds or data issues.

## Discord Alerts (Idempotent)

- Startup: Bot Starting (account, probe, feeds); Weekend Idle (Sat/Sun).
- Premarket: Premarket Start (universe), Risk Snapshot (cash, equity, BP, open risk vs cap, positions), Premarket Summary (bull/bear/neutral).
- [NEW] Trade: includes Order ID, Net Cost, Cash Remaining, BP Remaining, Slots Left.
- [FILLED]: order fill ping with fill price.
- [EXIT]: exit submitted (singles/spreads) with reason, entry vs exit (singles) or % of max + ROR (spreads), days.
- [CLOSED]: final fill with realized P/L and ROR.
- EOD: full synopsis — metrics, positions preview, attribution; messages deduped via outbox.

## Troubleshooting

- Options Probe shows Data API 404s: enable Options Market Data (indicative or OPRA) on your Alpaca key. Contracts discovery will still work via trading_v2 in auto mode.
- Snapshots EMPTY on weekend: expected; feed is indicative and markets are closed. Preflight is weekend-aware and will still report Ready if Contracts + Stream are OK.
- If preflight says Not Ready:
  - Check `.env` keys and `config.yaml` feeds.
  - Use `python -m src.app preflight` and read probe statuses in Discord.

## Notes

- Paper trading only. Do not enable live trading without adding an explicit config gate and thorough testing.
- Mid prices are used for slippage and valuation.
- Holidays are not accounted for yet; the scheduler skips weekends but not market holidays.
