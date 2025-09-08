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

### Intraday Entries (optional)

- By default, the bot can place entries intraday as well as during premarket.
- Control via `config.yaml` → `scheduling`:

  scheduling:
    intraday_every_seconds: 120
    intraday_entries_enabled: true       # enable/disable intraday entries
    intraday_entry_window: "09:40-15:30"  # ET window in which entries are allowed

    # Intraday Synopsis cadence (choose one)
    intraday_synopsis_every_minutes: 5     # preferred, in minutes
    # intraday_synopsis_every_seconds: 300 # alternative, in seconds

- Entries respect all the same liquidity, budget, and open-risk gates. Weekends are skipped automatically.

## Startup Summary + Plan

- On app start (even mid-day), the bot posts to the Trades webhook:
  - Market Summary (on start): Bullish | Bearish | Neutral counts via SMA20/50, plus a per-symbol bar count diagnostic (e.g., AAPL:150).
  - Plan (on start): Per-symbol plan. Default is lightweight (no options calls):
    `SYMBOL DIRECTION+MID -> STRATEGY [spot=$X.YZ]`.
  - To enable detailed startup plan (uses options contracts + snapshots to infer IV regime and expiry):

    diagnostics:
      startup_plan_mode: full   # lightweight | full

  - Detailed mode may take longer on slow networks; lightweight mode is fast and resilient.

### Broker Positions Sync (optional)

- On daemon start, the bot can import existing broker options positions into its local DB so the exit engine can manage them.
- Enabled by default; control via `config.yaml`:

  sync:
    broker_positions_on_start: true

- Notes:
  - Imports only long single‑leg CALL/PUT positions (to match current exit rules). Shorts and complex structures are skipped.
  - Entry debit is approximated from broker `avg_entry_price` if available.
  - Positions already present in the DB are not duplicated.

## How It Works

- Startup self-test posts to Discord:
  - Account snapshot, a Most-Actives sample, Options Probe (contracts endpoints + status), Options Stream auth result, Feeds in use.
  - Weekend Idle notice on Sat/Sun.
- Premarket (08:00 ET by default):
  - Universe build, trend signals (SMA20/50), options probe guard.
  - Announces Premarket Start (symbols) and a Risk Snapshot (cash, equity, buying power, open risk vs cap, positions).
  - For each symbol: discover contracts (prefers trading_v2), fetch snapshots (feed=indicative), map strategy, check liquidity and budgets, submit paper orders at mid; post [NEW].
  - Posts Premarket Summary (bull/bear/neutral counts and examples).
  - Posts Premarket Plan: a per-symbol rundown of intended entries with strikes/delta and estimated debit/width and cost.
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
- Liquidity: quotes must be present and the bid/ask spread <= cap; OI/Vol thresholds are enforced only if those fields are provided by the feed (indicative feed may omit them intraday).
- Strikes/Expiry: singles pick delta-band; spreads long ATM, short near implied move or fixed width; ensure net debit within per-trade budget.
  - Premarket Plan details per symbol:
    - CALL/PUT: `exp=YYYY-MM-DD K=<strike> Δ=<delta> mid=$<debit> qty=<n> est_cost=$<total>`
    - BULL_CALL_SPREAD/BEAR_PUT_SPREAD: `exp=YYYY-MM-DD Klong=<K1> Kshort=<K2> width=$<w> debit=$<debit> qty=<n> est_cost=$<total>`
  - If a candidate is skipped, the plan includes a concise reason (e.g., insufficient history, SMA not available, no contracts, illiquid, debit < min, budget too small, open-risk cap).

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
- Premarket: Premarket Start (universe), Risk Snapshot (cash, equity, BP, open risk vs cap, positions), Premarket Summary (bull/bear/neutral), Premarket Plan (per-symbol detailed plan).
- [NEW] Trade: includes Order ID, Net Cost, Cash Remaining, BP Remaining, Slots Left.
- [FILLED]: order fill ping with fill price.
- [EXIT]: exit submitted (singles/spreads) with reason, entry vs exit (singles) or % of max + ROR (spreads), days.
- [CLOSED]: final fill with realized P/L and ROR.
- EOD: full synopsis – metrics, positions preview, attribution; messages deduped via outbox.
- Intraday Synopsis: posts every ~5 minutes during market hours with per-symbol “eligible vs skipped (why)” using live data along with account snapshot (Open, Slots, Cash, BP, Open Risk vs Cap).

## Troubleshooting

- Options Probe shows Data API 404s: enable Options Market Data (indicative or OPRA) on your Alpaca key. Contracts discovery will still work via trading_v2 in auto mode.
- Snapshots EMPTY on weekend: expected; feed is indicative and markets are closed. Preflight is weekend-aware and will still report Ready if Contracts + Stream are OK.
- If preflight says Not Ready:
  - Check `.env` keys and `config.yaml` feeds.
  - Use `python -m src.app preflight` and read probe statuses in Discord.

### Not Enough Bars / Neutral Everything

- Ensure `signals.bars_lookback` is large enough (default 150). Bars are requested with explicit start/end and sorted by timestamp to ensure SMA alignment.
- Startup Market Summary includes a "Bars Returned (per symbol)" diagnostic field; symbols with <50 bars will be Neutral.

### Startup Plan Hangs

- The startup plan runs in "lightweight" mode by default (no options calls) to avoid blocking. If you set `diagnostics.startup_plan_mode: full` and see timeouts, switch back to lightweight or check network and options data entitlements.

## Notes

- Paper trading only. Do not enable live trading without adding an explicit config gate and thorough testing.
- Mid prices are used for slippage and valuation.
- Holidays are not accounted for yet; the scheduler skips weekends but not market holidays.
