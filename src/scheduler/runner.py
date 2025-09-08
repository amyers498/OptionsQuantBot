from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Any, Dict, List

from src.utils.logging import get_logger
from src.utils.time import now_et, today_et
from src.data.db import init_db, get_session
from src.data.repositories import OutboxRepo
from src.alpaca_client.auth import get_api_keys
from src.alpaca_client.market_data import MarketDataClient
from src.alpaca_client.options import OptionsClient
from src.alpaca_client.orders import OrdersClient
from src.alpaca_client.account import AccountClient
from src.alpaca_client.stream import probe_options_stream
from src.notify.discord import build_info_embed, build_trade_new_embed, send_webhook
from src.notify.discord import build_filled_embed, build_closed_embed
from src.pnl.valuers import legs_current_value, spread_current_value, spread_width_from_legs, parse_occ_expiry
from src.features.trend import compute_sma20_50
from src.features.liquidity import passes_liquidity
from src.features.implied_move import implied_move_abs
from src.features.iv_context import upsert_iv_summary
from src.selector.rules import SignalSnapshot, choose_strategy
from src.selector.strikes import Contract, pick_single_delta_band
from src.data.db import get_session
from src.data.models import Position, PositionStatus, PositionStructure


logger = get_logger(__name__)


def _build_clients(cfg: Dict[str, Any]):
    alp = cfg.get("alpaca", {})
    headers = get_api_keys()
    md = MarketDataClient(
        alp.get("base_url_paper", ""),
        alp.get("data_url", ""),
        headers,
        stocks_feed=alp.get("market_data_subscription", "iex"),
    )
    oc = OptionsClient(
        alp.get("data_url", ""),
        headers,
        trading_url=alp.get("base_url_paper", ""),
        options_feed=alp.get("options_feed", "indicative"),
        contracts_source=alp.get("contracts_source", "auto"),
    )
    orders = OrdersClient(alp.get("base_url_paper", ""), headers)
    acct = AccountClient(alp.get("base_url_paper", ""), headers)
    return headers, md, oc, orders, acct


def _send_once(webhook: str, key: str, payload: Dict[str, Any]) -> None:
    try:
        for session in get_session():
            repo = OutboxRepo(session)
            existing = repo.get(key)
            if existing and str(existing.status).upper() == 'SENT':
                return
            if not existing:
                repo.put(key, payload, status='PENDING')
            ok = send_webhook(webhook, payload)
            if ok:
                row = repo.get(key)
                if row:
                    import datetime as _dt
                    row.sent_at = _dt.datetime.utcnow()
                    row.status = 'SENT'
            break
    except Exception:
        # Fallback to direct send if DB fails
        send_webhook(webhook, payload)


def run_self_test(cfg: Dict[str, Any]) -> None:
    alp = cfg.get("alpaca", {})
    headers, md, oc, _, acct = _build_clients(cfg)
    disc = cfg.get("discord", {})
    webhook = disc.get("trades_webhook")
    color_neutral = int(disc.get("color_neutral", 15844367))

    fields: List[dict] = []
    try:
        a = acct.account()
        fields.append({"name": "Account", "value": f"cash=${a.get('cash')} equity=${a.get('equity')}", "inline": True})
    except Exception as e:
        fields.append({"name": "Account", "value": f"ERROR: {e}", "inline": True})

    try:
        syms = md.most_actives(top_n=5)
        fields.append({"name": "Most Actives", "value": ", ".join(syms) or "(none)", "inline": False})
    except Exception as e:
        fields.append({"name": "Most Actives", "value": f"ERROR: {e}", "inline": False})

    try:
        counts = oc.contracts_probe("AAPL", 7, 21)
        parts = []
        for k, v in counts.items():
            if isinstance(v, dict):
                parts.append(f"{k}: count={v.get('count')}, status={v.get('status')}")
            else:
                parts.append(f"{k}: {v}")
        fields.append({"name": "Options Probe", "value": "; ".join(parts), "inline": False})
    except Exception as e:
        fields.append({"name": "Options Probe", "value": f"ERROR: {e}", "inline": False})

    try:
        ok, info = probe_options_stream(alp.get("options_feed", "indicative"), headers)
        fields.append({"name": "Options Stream", "value": ("OK: " + info) if ok else ("ERROR: " + info), "inline": False})
    except Exception as e:
        fields.append({"name": "Options Stream", "value": f"ERROR: {e}", "inline": False})

    feeds_info = f"stocks_feed={alp.get('market_data_subscription','iex')}, options_feed={alp.get('options_feed','indicative')}"
    fields.append({"name": "Feeds", "value": feeds_info, "inline": False})

    if webhook:
        _send_once(webhook, f"SELFTEST:{today_et()}", build_info_embed(
            title="Bot Starting",
            description="Performing self-test (read-only probes)",
            color=color_neutral,
            fields=fields,
        ))

    # Deep readiness check → Ready/Not Ready embed
    ready_fields: List[dict] = []
    ready = True
    # Bars (non-blocking on weekends)
    try:
        bars = md.daily_bars("AAPL", lookback=10)
        ready_fields.append({"name": "Stocks Bars", "value": "OK" if bars else "EMPTY", "inline": True})
    except Exception as e:
        ready_fields.append({"name": "Stocks Bars", "value": f"ERROR: {e}", "inline": True})
    # Contracts + snapshots
    contracts_ok = False
    snapshots_ok = False
    try:
        cons = oc.contracts("AAPL", 7, 21)
        contracts_ok = bool(cons)
        ready_fields.append({"name": "Contracts", "value": f"OK ({len(cons)})" if contracts_ok else "EMPTY", "inline": True})
        symbols: List[str] = []
        for c in cons:
            if isinstance(c, str):
                symbols.append(c)
            elif isinstance(c, dict) and c.get("symbol"):
                symbols.append(c["symbol"]) 
            if len(symbols) >= 20:
                break
        if symbols:
            snaps = oc.snapshots(symbols)
            for sym in symbols:
                s = snaps.get(sym, {})
                q = s.get("latest_quote", {}) if isinstance(s, dict) else {}
                if isinstance(q, dict) and (q.get("bid_price") is not None or q.get("ask_price") is not None):
                    snapshots_ok = True
                    break
        ready_fields.append({"name": "Snapshots", "value": "OK" if snapshots_ok else "EMPTY", "inline": True})
    except Exception as e:
        ready_fields.append({"name": "Contracts/Snapshots", "value": f"ERROR: {e}", "inline": False})
        ready = False

    # Stream
    stream_ok = False
    try:
        s_ok, s_info = probe_options_stream(alp.get("options_feed", "indicative"), headers)
        stream_ok = bool(s_ok)
        ready_fields.append({"name": "Stream", "value": ("OK: " + s_info) if s_ok else ("ERROR: " + s_info), "inline": True})
    except Exception as e:
        ready_fields.append({"name": "Stream", "value": f"ERROR: {e}", "inline": True})

    # Weekend-aware gating: require Contracts + Stream; snapshots optional on weekend
    is_weekend = now_et().weekday() >= 5
    if is_weekend:
        ready = stream_ok and contracts_ok
        if not snapshots_ok:
            ready_fields.append({"name": "Weekend", "value": "Snapshots optional (market closed)", "inline": False})
    else:
        ready = stream_ok and contracts_ok and snapshots_ok

    if webhook:
        color_ok = int(disc.get("color_bullish", 3066993))
        color_err = int(disc.get("color_bearish", 15158332))
        send_webhook(webhook, build_info_embed(
            title="Ready to Trade" if ready else "Not Ready to Trade",
            description="All critical checks passed." if ready else "One or more checks failed.",
            color=(color_ok if ready else color_err),
            fields=ready_fields,
        ))

    # Weekend idle note
    if now_et().weekday() >= 5 and webhook:
        _send_once(webhook, f"WEEKEND:{today_et()}", build_info_embed(
            title="Weekend Idle",
            description="Market closed. The bot will idle until Monday while monitoring connectivity.",
            color=color_neutral,
        ))


def run_premarket(cfg: Dict[str, Any]) -> None:
    init_db()
    _, md, oc, orders, acct = _build_clients(cfg)
    disc = cfg.get("discord", {})
    webhook = disc.get("trades_webhook")
    color_neutral = int(disc.get("color_neutral", 15844367))

    # Universe
    uni_cfg = cfg.get("universe", {})
    top_n = int(uni_cfg.get("top_n", 15))
    exclude = uni_cfg.get("exclude", [])
    symbols = md.most_actives(top_n=top_n, exclude=exclude)
    if not symbols:
        static = uni_cfg.get("static", [])
        if static:
            symbols = [s for s in static if s not in exclude][:top_n]
    if webhook:
        _send_once(webhook, f"PREMARKET_START:{today_et()}", build_info_embed(
            title="Premarket Start",
            description=f"Universe size: {len(symbols)} | Top N requested: {top_n}",
            color=color_neutral,
            fields=[{"name": "Symbols", "value": ", ".join(symbols[:15]) or "(none)", "inline": False}],
        ))

    # Data availability guard (lightweight)
    try:
        probe = oc.contracts_probe("AAPL", 7, 21)
        has_data = any((isinstance(v, dict) and v.get("count", 0) > 0) for v in probe.values())
        if not has_data and webhook:
            _send_once(webhook, f"PREMARKET_SKIP:{today_et()}", build_info_embed(
                title="Premarket Skipped - Options Data Unavailable",
                description="Contracts probe returned 0 across endpoints.",
                color=color_neutral,
            ))
            return
    except Exception as e:
        logger.error({"event": "premarket_probe_error", "err": str(e)})

    # Trading logic
    sig_cfg = cfg.get("signals", {})
    sma_long = int(sig_cfg.get("sma_long", 50))
    lookback_iv_days = int(sig_cfg.get("iv_percentile_lookback_days", 90))
    lo_thr = float(sig_cfg.get("iv_low_threshold_pctile", 30))
    hi_thr = float(sig_cfg.get("iv_high_threshold_pctile", 70))

    strat_cfg = cfg.get("strategy", {})
    dmin = int(strat_cfg.get("expiry_min_days", 7))
    dmax = int(strat_cfg.get("expiry_max_days", 21))
    d_lo = float(strat_cfg.get("single_delta_min", 0.35))
    d_hi = float(strat_cfg.get("single_delta_max", 0.55))
    width_default = float(strat_cfg.get("spread_width_default", 5.0))
    use_im_cap = bool(strat_cfg.get("use_implied_move_for_cap", True))

    liq = cfg.get("liquidity", {})
    min_oi = int(liq.get("min_open_interest", 500))
    min_vol = int(liq.get("min_volume", 50))
    max_spread_pct = float(liq.get("max_spread_pct", 0.10))

    risk_cfg = cfg.get("risk", {})
    per_min = float(risk_cfg.get("per_trade_dollar_min", 50))
    per_max = float(risk_cfg.get("per_trade_dollar_max", 200))
    min_premium = float(risk_cfg.get("min_premium", 0.4))
    max_positions = int(risk_cfg.get("max_concurrent_positions", 4))

    # Account snapshot
    def _to_float(v, default=0.0):
        try:
            return default if v is None else float(v)
        except Exception:
            return default
    try:
        a = acct.account()
    except Exception:
        a = {}
    cash = _to_float(a.get("cash"), 0.0)
    equity = _to_float(a.get("equity"), cash)
    buying_power = _to_float(a.get("buying_power"), cash)
    try:
        broker_positions = acct.positions()
    except Exception:
        broker_positions = []
    def _is_opt(p: dict) -> bool:
        return (str(p.get("asset_class") or "").lower() == "option") or len(str(p.get("symbol") or "")) >= 15
    current_positions = len([p for p in broker_positions if isinstance(p, dict) and _is_opt(p)])

    bull_syms: List[str] = []
    bear_syms: List[str] = []
    neutral_syms: List[str] = []
    placed_count = 0
    spent_budget = 0.0
    # Strict open-risk cap (sum of entry_debit for OPEN positions)
    # Compute once at start; update as we place entries
    open_risk_cap_abs = (float(cfg.get("risk", {}).get("max_open_risk_pct_of_equity", 40)) / 100.0) * max(equity, 0.0)
    try:
        for session in get_session():
            from sqlalchemy import func as _func
            open_risk_now = session.query(_func.coalesce(_func.sum(Position.entry_debit), 0.0)).filter(Position.status == PositionStatus.OPEN).scalar()
            open_risk_sum = float(open_risk_now or 0.0)
            break
    except Exception:
        open_risk_sum = 0.0

    import pandas as pd

    for sym in symbols:
        try:
            if current_positions + placed_count >= max_positions:
                break
            # Trend via SMA20/50
            bars = md.daily_bars(sym, lookback=max(sma_long + 5, 60))
            closes = [b.get("close") for b in bars if b.get("close") is not None]
            if len(closes) < sma_long:
                continue
            df = compute_sma20_50(pd.DataFrame({"close": closes}))
            sma20 = float(df["sma20"].iloc[-1]) if pd.notnull(df["sma20"].iloc[-1]) else None
            sma50 = float(df["sma50"].iloc[-1]) if pd.notnull(df["sma50"].iloc[-1]) else None
            if sma20 is None or sma50 is None:
                continue
            direction = "BULL" if sma20 > sma50 else ("BEAR" if sma20 < sma50 else "NEUTRAL")
            if direction == "NEUTRAL":
                neutral_syms.append(sym)
                continue
            (bull_syms if direction == "BULL" else bear_syms).append(sym)

            # Spot
            spot = md.latest_trade_price(sym)
            if spot is None:
                continue
            # Contracts near DTE window
            cons = oc.contracts(sym, dmin, dmax)
            if not cons:
                # Fallback: extend window to 28 days
                cons = oc.contracts(sym, dmin, 28)
            cons = [c for c in cons if c.get("expiration_date")]
            if not cons:
                continue
            from datetime import datetime as _dt, date as _date
            def _dte(exp: str) -> int:
                try:
                    d = _dt.fromisoformat(exp).date()
                except Exception:
                    d = _date.fromisoformat(exp)
                return (d - _date.today()).days
            cons.sort(key=lambda c: abs(_dte(c["expiration_date"])) )
            expiry = cons[0]["expiration_date"]
            same_exp = [c for c in cons if c.get("expiration_date") == expiry]
            same_exp.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
            focus = same_exp[:30]
            occ_syms = [c.get("symbol") for c in focus if c.get("symbol")]
            snaps = oc.snapshots(occ_syms)

            # Helpers
            def _mid(occ: str) -> float | None:
                s = snaps.get(occ, {})
                q = s.get("latest_quote", {})
                bp, ap = q.get("bid_price"), q.get("ask_price")
                if not bp or not ap or bp <= 0 or ap <= 0:
                    return None
                return (bp + ap) / 2.0
            def _iv(occ: str) -> float | None:
                s = snaps.get(occ, {}); g = s.get("greeks", {}); v = g.get("iv"); return float(v) if v is not None else None

            calls = [c for c in focus if c.get("type") == "call"]
            puts = [c for c in focus if c.get("type") == "put"]
            calls.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
            puts.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
            atm_call = calls[0] if calls else None
            atm_put = puts[0] if puts else None
            if not atm_call or not atm_put:
                continue
            atm_call_mid = _mid(atm_call["symbol"]) if atm_call else None
            atm_put_mid = _mid(atm_put["symbol"]) if atm_put else None

            iv_vals = [v for v in (_iv(atm_call["symbol"]), _iv(atm_put["symbol"])) if v is not None]
            atm_iv = sum(iv_vals) / len(iv_vals) if iv_vals else None
            # Store IV percentile
            pct = None
            try:
                for session in get_session():
                    _, pct = upsert_iv_summary(session, sym, today_et(), float(atm_call.get("strike_price")) if atm_call else None,
                                               _dt.fromisoformat(expiry).date() if isinstance(expiry, str) else expiry,
                                               atm_iv, lookback_iv_days)
                    break
            except Exception:
                pct = None
            iv_regime = "MID" if pct is None else ("LOW" if pct <= lo_thr else ("HIGH" if pct >= hi_thr else "MID"))

            strat = choose_strategy(SignalSnapshot(direction=direction, iv_regime=iv_regime))
            if strat is None:
                continue

            # Liquidity baseline
            def _liq(occ: str, eff_spread_cap: float | None = None) -> bool:
                s = snaps.get(occ, {}); q = s.get("latest_quote", {})
                oi = s.get("open_interest") or 0; vol = s.get("day", {}).get("volume") or 0
                cap = max_spread_pct if eff_spread_cap is None else eff_spread_cap
                return oi >= min_oi and vol >= min_vol and passes_liquidity(oi, vol, q.get("bid_price"), q.get("ask_price"), cap)
            if not (_liq(atm_call["symbol"]) and _liq(atm_put["symbol"])):
                continue

            order_color = int(disc.get("color_bullish", 3066993)) if direction == "BULL" else int(disc.get("color_bearish", 15158332))
            title = f"[NEW] {sym} {strat} ({expiry})"
            desc = f"Strategy: {strat}; Reason: {direction}+{iv_regime}"

            legs: List[Dict[str, Any]] = []
            limit_price = 0.0
            net_cost = 0.0

            if strat in ("CALL", "PUT"):
                side_list = calls if strat == "CALL" else puts
                cands: List[Contract] = []
                for c in side_list:
                    s = snaps.get(c["symbol"], {}); g = s.get("greeks", {}); q = s.get("latest_quote", {})
                    cands.append(Contract(symbol=c["symbol"], expiry=expiry, strike=float(c.get("strike_price", 0.0)),
                                          type=("C" if c.get("type") == "call" else "P"), delta=g.get("delta"),
                                          bid=q.get("bid_price"), ask=q.get("ask_price"), mid=_mid(c["symbol"]) or 0.0))
                pick = pick_single_delta_band(cands, d_lo, d_hi)
                if not pick:
                    continue
                s = snaps.get(pick.symbol, {}); q = s.get("latest_quote", {})
                oi = s.get("open_interest") or 0; vol = s.get("day", {}).get("volume") or 0
                # Dynamic spread cap: tighten to 5% when premium < $0.80
                eff_cap = 0.05 if (pick.mid or 0.0) < 0.80 else max_spread_pct
                if not passes_liquidity(oi, vol, q.get("bid_price"), q.get("ask_price"), eff_cap):
                    continue
                debit = pick.mid or 0.0
                if debit <= 0 or debit < min_premium:
                    continue
                qty = int(max(1, per_max // (debit * 100)))
                if qty < 1 or (debit * 100) < per_min:
                    continue
                legs = [{"symbol": pick.symbol, "side": "buy", "qty": qty}]
                limit_price = round(debit, 2)
                net_cost = debit * 100.0 * qty
                # Strict open-risk cap gate
                if (open_risk_sum + net_cost) > open_risk_cap_abs:
                    # skip due to open-risk cap
                    continue
            else:
                if atm_call_mid is None or atm_put_mid is None:
                    continue
                im = implied_move_abs(atm_call_mid, atm_put_mid) or width_default
                if direction == "BULL":
                    long_leg = atm_call; target = spot + (im if use_im_cap else width_default); short_pool = calls
                else:
                    long_leg = atm_put; target = spot - (im if use_im_cap else width_default); short_pool = puts
                short_pool.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - target))
                short_leg = short_pool[0] if short_pool else None
                def _liq_leg(c: dict) -> bool:
                    s = snaps.get(c["symbol"], {}); q = s.get("latest_quote", {})
                    return passes_liquidity(s.get("open_interest") or 0, s.get("day", {}).get("volume") or 0, q.get("bid_price"), q.get("ask_price"), max_spread_pct)
                if not short_leg or not _liq_leg(short_leg):
                    alt = spot + (width_default if direction == "BULL" else -width_default)
                    short_pool.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - alt))
                    if short_pool and _liq_leg(short_pool[0]):
                        short_leg = short_pool[0]
                if not short_leg:
                    continue
                long_mid = _mid(long_leg["symbol"]) or 0.0
                short_mid = _mid(short_leg["symbol"]) or 0.0
                if long_mid <= 0 or short_mid <= 0:
                    continue
                net_debit = max(0.0, long_mid - short_mid)
                if net_debit < min_premium or not (per_min <= net_debit * 100 <= per_max):
                    continue
                qty = 1
                legs = [
                    {"symbol": long_leg["symbol"], "side": "buy", "qty": qty},
                    {"symbol": short_leg["symbol"], "side": "sell", "qty": qty},
                ]
                limit_price = round(net_debit, 2)
                net_cost = net_debit * 100.0 * qty
                # Strict open-risk cap gate
                if (open_risk_sum + net_cost) > open_risk_cap_abs:
                    continue

            # Risk checks vs cash/BP and positions
            available_cash = max(0.0, cash - spent_budget)
            if net_cost > available_cash or net_cost > buying_power:
                continue

            # Submit
            try:
                resp = orders.submit_options_order(legs=legs, limit_price=limit_price)
            except Exception as e:
                logger.error({"event": "order_submit_error", "symbol": sym, "err": str(e)})
                continue
            order_id = resp.get("id", "")
            placed_count += 1
            spent_budget += net_cost
            open_risk_sum += net_cost

            # Persist minimal position record for exit engine
            try:
                for session in get_session():
                    structure = (
                        PositionStructure.CALL if strat == "CALL" else
                        PositionStructure.PUT if strat == "PUT" else
                        PositionStructure.BULL_CALL_SPREAD if strat == "BULL_CALL_SPREAD" else
                        PositionStructure.BEAR_PUT_SPREAD
                    )
                    pos = Position(
                        ticker=sym,
                        structure=structure,
                        legs_json=legs,
                        opened_at=now_et(),
                        status=PositionStatus.OPEN,
                        entry_debit=net_cost,
                        notes=f"dir={direction}; iv_regime={iv_regime}; expiry={expiry}",
                    )
                    session.add(pos)
                    break
            except Exception as e:
                logger.error({"event": "persist_position_error", "symbol": sym, "err": str(e)})

            # Discord [NEW]
            if webhook:
                cash_after = max(0.0, cash - spent_budget)
                bp_after = max(0.0, buying_power - net_cost)
                slots_left = max(0, max_positions - (current_positions + placed_count))
                _send_once(webhook, f"NEW:{order_id}", build_trade_new_embed(
                    title=title,
                    description=desc,
                    color=order_color,
                    fields=[
                        {"name": "Order ID", "value": order_id or "(pending)", "inline": True},
                        {"name": "Net Cost", "value": f"${net_cost:,.2f}", "inline": True},
                        {"name": "Cash Remaining", "value": f"${cash_after:,.2f}", "inline": True},
                        {"name": "BP Remaining", "value": f"${bp_after:,.2f}", "inline": True},
                        {"name": "Slots Left", "value": str(slots_left), "inline": True},
                    ],
                ))

            # Optional: short poll for fills to send [FILLED]
            try:
                od = orders.wait_for_status(order_id, ["filled", "partially_filled"], timeout_sec=5.0)
                if od and webhook:
                    fill_price = od.get("filled_avg_price") or od.get("limit_price")
                    _send_once(webhook, f"FILLED:{order_id}", build_filled_embed(
                        title=f"[FILLED] {sym}",
                        fields=[
                            {"name": "Order ID", "value": order_id, "inline": True},
                            {"name": "Fill", "value": f"${fill_price}", "inline": True},
                            {"name": "Qty", "value": str(legs[0].get("qty", 1)), "inline": True},
                        ],
                        color=order_color,
                    ))
            except Exception:
                pass
        except Exception as e:
            logger.error({"event": "premarket_error", "symbol": sym, "err": str(e)})

    # Summary
    if webhook:
        _send_once(webhook, f"PREMARKET_SUMMARY:{today_et()}", build_info_embed(
            title="Premarket Summary",
            description=f"Bullish={len(bull_syms)} | Bearish={len(bear_syms)} | Neutral={len(neutral_syms)}",
            color=color_neutral,
        ))


def run_intraday(cfg: Dict[str, Any]) -> None:
    logger.info({"event": "intraday_tick"})
    headers, md, oc, orders, _ = _build_clients(cfg)
    disc = cfg.get("discord", {})
    webhook = disc.get("trades_webhook")
    color_neutral = int(disc.get("color_neutral", 15844367))

    exits_cfg = cfg.get("exits", {})
    single_tp_lo = float(exits_cfg.get("single_profit_target_lo", 0.5))
    single_tp_hi = float(exits_cfg.get("single_profit_target_hi", 1.0))
    single_hard_stop = float(exits_cfg.get("single_hard_stop_pct", 0.5))
    time_stop_max = int(exits_cfg.get("time_stop_days_max", 5))

    # Load open positions
    try:
        for session in get_session():
            open_positions = session.query(Position).filter(Position.status == PositionStatus.OPEN).all()
            break
    except Exception as e:
        logger.error({"event": "intraday_positions_error", "err": str(e)})
        return
    if not open_positions:
        return

    # Collect OCC symbols for singles
    occs: List[str] = []
    for p in open_positions:
        if p.structure in (PositionStructure.CALL, PositionStructure.PUT):
            try:
                occs.append(p.legs_json[0]["symbol"])
            except Exception:
                continue
    snaps = oc.snapshots(occs) if occs else {}

    # Evaluate singles exits
    for p in open_positions:
        if p.structure not in (PositionStructure.CALL, PositionStructure.PUT):
            continue
        try:
            leg = p.legs_json[0]
            occ = leg.get("symbol")
            qty = int(leg.get("qty", 1))
            side = str(leg.get("side", "buy")).lower()
            s = snaps.get(occ, {})
            q = s.get("latest_quote", {}) if isinstance(s, dict) else {}
            bid = q.get("bid_price"); ask = q.get("ask_price")
            mid = ((bid or 0) + (ask or 0)) / 2.0 if bid and ask else None
            if not mid or mid <= 0:
                continue
            entry = float(p.entry_debit or 0.0)
            if entry <= 0:
                continue
            current_val = mid * 100.0 * qty
            roi = (current_val - entry) / entry
            days_held = max(0, (now_et().date() - p.opened_at.date()).days)

            exit_reason = None
            if side == "buy":
                if roi >= single_tp_hi:
                    exit_reason = f"TP {int(single_tp_hi*100)}%"
                elif roi >= single_tp_lo:
                    exit_reason = f"TP {int(single_tp_lo*100)}%"
                elif roi <= -abs(single_hard_stop):
                    exit_reason = f"Hard Stop {int(single_hard_stop*100)}%"
            if not exit_reason and days_held >= time_stop_max:
                exit_reason = f"Time {days_held}d"

            # Optional: signal flip
            if not exit_reason:
                try:
                    bars = md.daily_bars(p.ticker, lookback=60)
                    import pandas as pd
                    closes = [b.get("close") for b in bars if b.get("close") is not None]
                    if len(closes) >= 50:
                        df = compute_sma20_50(pd.DataFrame({"close": closes}))
                        sma20 = float(df["sma20"].iloc[-1]) if pd.notnull(df["sma20"].iloc[-1]) else None
                        sma50 = float(df["sma50"].iloc[-1]) if pd.notnull(df["sma50"].iloc[-1]) else None
                        if sma20 is not None and sma50 is not None:
                            dir_now = "BULL" if sma20 > sma50 else ("BEAR" if sma20 < sma50 else "NEUTRAL")
                            dir_entry = "NEUTRAL"
                            if p.notes and "dir=" in p.notes:
                                try:
                                    dir_entry = p.notes.split("dir=")[1].split(";")[0]
                                except Exception:
                                    pass
                            if dir_now != dir_entry and dir_now != "NEUTRAL":
                                exit_reason = f"Signal Flip {dir_entry}->{dir_now}"
                except Exception as e:
                    logger.error({"event": "signal_flip_error", "ticker": p.ticker, "err": str(e)})

            if not exit_reason:
                continue

            # Submit close order for singles
            close_legs = [{"symbol": occ, "side": "sell", "qty": qty, "position_intent": "sell_to_close"}]
            try:
                resp = orders.submit_options_order(legs=close_legs, limit_price=round(mid, 2))
                order_id = resp.get("id", "")
            except Exception as e:
                logger.error({"event": "exit_order_error", "symbol": occ, "err": str(e)})
                continue

            # Mark position closed
            try:
                for session in get_session():
                    dbp = session.get(Position, p.id)
                    if dbp:
                        dbp.status = PositionStatus.CLOSED
                    break
            except Exception as e:
                logger.error({"event": "db_close_error", "pos_id": p.id, "err": str(e)})

            # Discord EXIT
            if webhook:
                fields = [
                    {"name": "Reason", "value": exit_reason, "inline": False},
                    {"name": "Entry Debit", "value": f"${entry:,.2f}", "inline": True},
                    {"name": "Exit Value", "value": f"${current_val:,.2f}", "inline": True},
                    {"name": "ROI", "value": f"{roi*100:.1f}%", "inline": True},
                    {"name": "Days Held", "value": str(days_held), "inline": True},
                ]
                _send_once(webhook, f"EXIT:{p.id}:{occ}", build_info_embed(
                    title=f"[EXIT] {p.ticker} {p.structure.name}",
                    description=f"Close order submitted ({occ})",
                    color=color_neutral,
                    fields=fields,
                ))

            # Wait for close fill/cancel policy
            try:
                od = orders.wait_for_status(order_id, ["filled"], timeout_sec=45.0)
                if not od:
                    # cancel remainder
                    try:
                        orders.cancel_order(order_id)
                    except Exception:
                        pass
                # Post [CLOSED]
                fill_price = (od or {}).get("filled_avg_price") or round(mid, 2)
                realized = current_val - entry
                ror = (current_val - entry) / entry if entry > 0 else 0.0
                # persist closed_value/closed_at
                try:
                    for session in get_session():
                        dbp = session.get(Position, p.id)
                        if dbp:
                            from datetime import datetime as _dt
                            dbp.closed_value = current_val
                            dbp.closed_at = now_et()
                        break
                except Exception:
                    pass
                if webhook:
                    _send_once(webhook, f"CLOSED:{p.id}:{occ}", build_closed_embed(
                        title=f"[CLOSED] {p.ticker} {p.structure.name}",
                        description=exit_reason,
                        fields=[
                            {"name": "Entry Debit", "value": f"${entry:,.2f}", "inline": True},
                            {"name": "Exit Value", "value": f"${current_val:,.2f}", "inline": True},
                            {"name": "Realized P/L", "value": f"${realized:,.2f}", "inline": True},
                            {"name": "ROR", "value": f"{ror*100:.1f}%", "inline": True},
                            {"name": "Days Held", "value": str(days_held), "inline": True},
                        ],
                    ))
            except Exception:
                pass
        except Exception as e:
            logger.error({"event": "intraday_exit_error", "pos_id": getattr(p, 'id', None), "err": str(e)})

    # Evaluate spread exits (BULL_CALL_SPREAD, BEAR_PUT_SPREAD)
    spread_positions = [p for p in open_positions if p.structure in (PositionStructure.BULL_CALL_SPREAD, PositionStructure.BEAR_PUT_SPREAD)]
    # Build OCC list for all spread legs
    spread_occs: List[str] = []
    for p in spread_positions:
        for leg in (p.legs_json or []):
            if isinstance(leg, dict) and leg.get("symbol"):
                spread_occs.append(leg["symbol"])
    spread_snaps = oc.snapshots(spread_occs) if spread_occs else {}

    pt_pct = float(cfg.get("exits", {}).get("spread_profit_target_pct_of_max", 0.7))

    from src.pnl.valuers import spread_current_value, spread_width_from_legs

    for p in spread_positions:
        try:
            legs = p.legs_json or []
            # Compute current value
            curr_val = spread_current_value(legs, spread_snaps)
            # Compute max value using strikes and qty (assume both legs same qty)
            width = spread_width_from_legs(legs) or 0.0
            qty = int(legs[0].get("qty", 1)) if legs and isinstance(legs[0], dict) else 1
            max_val = max(0.0, width) * 100.0 * qty
            if max_val <= 0:
                continue
            entry = float(p.entry_debit or 0.0)
            roi = (curr_val - entry) / entry if entry > 0 else 0.0
            pct_to_max = curr_val / max_val
            days_held = max(0, (now_et().date() - p.opened_at.date()).days)

            exit_reason = None
            # Profit target
            if pct_to_max >= pt_pct:
                exit_reason = f"PT_SPREAD {int(pt_pct*100)}% of max"
            # Hard stop: -50% of debit (curr <= 0.5 * entry)
            elif entry > 0 and curr_val <= (entry * (1.0 - float(cfg.get("exits", {}).get("single_hard_stop_pct", 0.5)))):
                exit_reason = "HS_SPREAD -50%"
                exit_reason = "HS_SPREAD -50%"
            # Time stop
            elif days_held >= int(cfg.get("exits", {}).get("time_stop_days_max", 5)):
                exit_reason = f"TIME_STOP {days_held}d"
            # Signal flip
            if not exit_reason:
                try:
                    bars = md.daily_bars(p.ticker, lookback=60)
                    import pandas as pd
                    closes = [b.get("close") for b in bars if b.get("close") is not None]
                    if len(closes) >= 50:
                        df = compute_sma20_50(pd.DataFrame({"close": closes}))
                        sma20 = float(df["sma20"].iloc[-1]) if pd.notnull(df["sma20"].iloc[-1]) else None
                        sma50 = float(df["sma50"].iloc[-1]) if pd.notnull(df["sma50"].iloc[-1]) else None
                        if sma20 is not None and sma50 is not None:
                            dir_now = "BULL" if sma20 > sma50 else ("BEAR" if sma20 < sma50 else "NEUTRAL")
                            dir_entry = None
                            if p.notes and "dir=" in p.notes:
                                try:
                                    dir_entry = p.notes.split("dir=")[1].split(";")[0]
                                except Exception:
                                    pass
                            if dir_entry and dir_now != dir_entry and dir_now != "NEUTRAL":
                                exit_reason = f"SIGNAL_FLIP {dir_entry}->{dir_now}"
                except Exception as e:
                    logger.error({"event": "spread_signal_flip_error", "ticker": p.ticker, "err": str(e)})

            if not exit_reason:
                continue

            # Build close legs: reverse intents
            close_legs: List[Dict[str, Any]] = []
            for leg in legs:
                sym = leg.get("symbol"); q = int(leg.get("qty", 1))
                side = str(leg.get("side", "buy")).lower()
                if side == "buy":
                    close_legs.append({"symbol": sym, "side": "sell", "qty": q, "position_intent": "sell_to_close"})
                else:
                    close_legs.append({"symbol": sym, "side": "buy", "qty": q, "position_intent": "buy_to_close"})

            # Use current mid (per-leg) to set a conservative limit: curr_val per 100 per qty
            limit_price = round(curr_val / (100.0 * qty), 2) if qty > 0 else 0.01
            try:
                resp = orders.submit_options_order(legs=close_legs, limit_price=limit_price)
                order_id = resp.get("id", "")
            except Exception as e:
                logger.error({"event": "spread_exit_order_error", "symbol": p.ticker, "err": str(e)})
                continue

            # Mark closed
            try:
                for session in get_session():
                    dbp = session.get(Position, p.id)
                    if dbp:
                        dbp.status = PositionStatus.CLOSED
                    break
            except Exception as e:
                logger.error({"event": "db_close_error", "pos_id": p.id, "err": str(e)})

            # Discord EXIT
            if webhook:
                _send_once(webhook, f"EXIT:{p.id}", build_info_embed(
                    title=f"[EXIT] {p.ticker} {p.structure.name}",
                    description=f"{exit_reason}",
                    color=color_neutral,
                    fields=[
                        {"name": "% of Max", "value": f"{pct_to_max*100:.1f}%", "inline": True},
                        {"name": "ROR", "value": f"{roi*100:.1f}%", "inline": True},
                        {"name": "Days Held", "value": str(days_held), "inline": True},
                    ],
                ))

            # Wait for close fill/cancel, then [CLOSED]
            try:
                od = orders.wait_for_status(order_id, ["filled"], timeout_sec=45.0)
                if not od:
                    try:
                        orders.cancel_order(order_id)
                    except Exception:
                        pass
                realized = curr_val - entry
                ror = (curr_val - entry) / entry if entry > 0 else 0.0
                try:
                    for session in get_session():
                        dbp = session.get(Position, p.id)
                        if dbp:
                            dbp.closed_value = curr_val
                            dbp.closed_at = now_et()
                        break
                except Exception:
                    pass
                if webhook:
                    _send_once(webhook, f"CLOSED:{p.id}", build_closed_embed(
                        title=f"[CLOSED] {p.ticker} {p.structure.name}",
                        description=exit_reason,
                        fields=[
                            {"name": "% of Max", "value": f"{pct_to_max*100:.1f}%", "inline": True},
                            {"name": "Realized P/L", "value": f"${realized:,.2f}", "inline": True},
                            {"name": "ROR", "value": f"{ror*100:.1f}%", "inline": True},
                            {"name": "Days Held", "value": str(days_held), "inline": True},
                        ],
                    ))
            except Exception:
                pass

        except Exception as e:
            logger.error({"event": "spread_exit_error", "pos_id": getattr(p, 'id', None), "err": str(e)})

def run_eod(cfg: Dict[str, Any]) -> None:
    headers, md, oc, orders, acct = _build_clients(cfg)
    disc = cfg.get("discord", {})
    eod = disc.get("eod_webhook")
    try:
        a = acct.account()
        cash = float(a.get('cash') or 0.0)
        equity = float(a.get('equity') or 0.0)
        # Load positions
        for session in get_session():
            all_positions = session.query(Position).all()
            break
        # Snapshots for open positions
        open_positions = [p for p in all_positions if p.status == PositionStatus.OPEN]
        occs: List[str] = []
        for p in open_positions:
            for leg in (p.legs_json or []):
                if isinstance(leg, dict) and leg.get("symbol"):
                    occs.append(leg["symbol"])
        snaps = oc.snapshots(occs) if occs else {}

        # Compute P&L
        unreal = 0.0
        open_risk = 0.0
        greeks = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
        lines: List[str] = []
        today = today_et()
        realized = 0.0
        # Attribution buckets
        buckets: Dict[str, float] = {}

        # Open positions valuation and Greeks
        for p in open_positions:
            legs = p.legs_json or []
            curr_val = legs_current_value(legs, snaps)
            open_risk += float(p.entry_debit or 0.0)
            unreal += (curr_val - float(p.entry_debit or 0.0))
            # Greeks rollup
            net_d = net_t = net_v = 0.0
            for leg in legs:
                occ = leg.get("symbol")
                qty = int(leg.get("qty", 1))
                side = str(leg.get("side", "buy")).lower()
                snap = snaps.get(occ, {})
                g = snap.get("greeks", {})
                sgn = 1 if side == 'buy' else -1
                if g:
                    net_d += sgn * float(g.get("delta") or 0.0) * qty
                    net_t += sgn * float(g.get("theta") or 0.0) * qty
                    net_v += sgn * float(g.get("vega") or 0.0) * qty
            greeks["delta"] += net_d
            greeks["theta"] += net_t
            greeks["vega"] += net_v
            # DTE and details line
            dtes = []
            for leg in legs:
                from src.pnl.valuers import parse_occ_expiry
                d = parse_occ_expiry(leg.get("symbol", ""))
                if d:
                    dtes.append((d - today).days)
            dte_min = min(dtes) if dtes else 0
            # Strikes summary
            strikes = []
            for leg in legs:
                from src.pnl.valuers import parse_occ_strike
                k = parse_occ_strike(leg.get("symbol", ""))
                if k is not None:
                    strikes.append(k)
            lines.append(f"{p.ticker} | {p.structure.name} | cost ${p.entry_debit:.2f} | mark ${curr_val:.2f} | UPL ${curr_val - (p.entry_debit or 0):.2f} | DTE {dte_min}")

        # Realized for positions closed today (based on closed_value)
        for p in all_positions:
            if p.status == PositionStatus.CLOSED and p.closed_at and p.closed_at.date() == today:
                rv = float(p.closed_value or 0.0)
                realized += (rv - float(p.entry_debit or 0.0))
                # Attribution bucket by notes
                bucket = "UNKNOWN"
                if p.notes:
                    try:
                        dir_part = p.notes.split("dir=")[1].split(";")[0]
                        iv_part = p.notes.split("iv_regime=")[1].split(";")[0]
                        bucket = f"{dir_part}+{iv_part}+{p.structure.name}"
                    except Exception:
                        pass
                buckets[bucket] = buckets.get(bucket, 0.0) + (rv - float(p.entry_debit or 0.0))

        total_pl = realized + unreal

        if eod:
            # Metrics embed
            fields = [
                {"name": "Cash", "value": f"${cash:,.2f}", "inline": True},
                {"name": "Equity", "value": f"${equity:,.2f}", "inline": True},
                {"name": "Realized (day)", "value": f"${realized:,.2f}", "inline": True},
                {"name": "Unrealized", "value": f"${unreal:,.2f}", "inline": True},
                {"name": "Total P/L", "value": f"${total_pl:,.2f}", "inline": True},
                {"name": "Open Risk", "value": f"${open_risk:,.2f}", "inline": True},
                {"name": "Greeks", "value": f"Δ {greeks['delta']:.2f} Θ {greeks['theta']:.2f} ν {greeks['vega']:.2f}", "inline": False},
            ]
            _send_once(eod, f"EOD:{today}", {
                "embeds": [{
                    "title": "EOD Summary",
                    "description": "Account + Portfolio Metrics",
                    "fields": fields,
                }]
            })

            # Positions table (first few lines)
            if lines:
                preview = "\n".join(lines[:10])
                _send_once(eod, f"EOD_POS:{today}", {
                    "embeds": [{
                        "title": "Positions (preview)",
                        "description": f"```\n{preview}\n```",
                    }]
                })

            # Attribution
            if buckets:
                bucket_lines = [f"{k}: ${v:,.2f}" for k, v in list(buckets.items())[:10]]
                _send_once(eod, f"EOD_ATTR:{today}", {
                    "embeds": [{
                        "title": "Attribution (P/L by bucket)",
                        "description": "\n".join(bucket_lines),
                    }]
                })
    except Exception as e:
        logger.error({"event": "eod_error", "err": str(e)})


def _parse_cron_min_hour(expr: str) -> tuple[int, int] | None:
    try:
        parts = (expr or "").split()
        if len(parts) < 2:
            return None
        m = int(parts[0]); h = int(parts[1])
        if 0 <= m <= 59 and 0 <= h <= 23:
            return m, h
    except Exception:
        return None
    return None


def _et_between(now: datetime, start: dtime, end: dtime) -> bool:
    return start <= now.time() < end


def run_daemon(cfg: Dict[str, Any]) -> None:
    init_db()
    sched = cfg.get("scheduling", {})
    pre_mh = _parse_cron_min_hour(sched.get("premarket_cron", "0 8 * * 1-5")) or (0, 8)
    eod_mh = _parse_cron_min_hour(sched.get("eod_cron", "15 16 * * 1-5")) or (15, 16)
    intraday_every = int(sched.get("intraday_every_seconds", 120))

    pre_start = dtime(hour=pre_mh[1], minute=pre_mh[0])
    market_open = dtime(hour=9, minute=30)
    eod_time = dtime(hour=eod_mh[1], minute=eod_mh[0])
    market_close_guard = dtime(hour=18, minute=0)

    last_intraday_ts: float | None = None
    premarket_done_for: str | None = None
    eod_done_for: str | None = None

    logger.info({
        "event": "daemon_start",
        "premarket_start": str(pre_start),
        "market_open": str(market_open),
        "eod_time": str(eod_time),
        "intraday_every": intraday_every,
    })

    try:
        while True:
            now = now_et()
            today_key = str(today_et())

            if premarket_done_for and premarket_done_for != today_key:
                premarket_done_for = None
            if eod_done_for and eod_done_for != today_key:
                eod_done_for = None

            if now.weekday() >= 5:
                time.sleep(60)
                continue

            if _et_between(now, pre_start, market_open):
                if premarket_done_for != today_key:
                    logger.info({"event": "daemon_phase", "phase": "premarket"})
                    try:
                        run_premarket(cfg)
                    finally:
                        premarket_done_for = today_key
                time.sleep(10)
                continue

            if _et_between(now, market_open, eod_time):
                ts = now.timestamp()
                if last_intraday_ts is None or (ts - last_intraday_ts) >= intraday_every:
                    logger.info({"event": "daemon_phase", "phase": "intraday"})
                    try:
                        run_intraday(cfg)
                    finally:
                        last_intraday_ts = ts
                time.sleep(1)
                continue

            if now.time() >= eod_time and now.time() < market_close_guard:
                if eod_done_for != today_key:
                    logger.info({"event": "daemon_phase", "phase": "eod"})
                    try:
                        run_eod(cfg)
                    finally:
                        eod_done_for = today_key
                time.sleep(30)
                continue

            time.sleep(60)
    except KeyboardInterrupt:
        logger.info({"event": "daemon_stop", "reason": "KeyboardInterrupt"})
        try:
            run_eod(cfg)
        except Exception as e:
            logger.error({"event": "daemon_stop_eod_error", "err": str(e)})
