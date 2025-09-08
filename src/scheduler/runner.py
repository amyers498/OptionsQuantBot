from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Any, Dict, List
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.time import now_et, today_et, set_local_tz
from src.utils.config import load_config
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


def _format_money(v: float | None) -> str:
    try:
        return f"${float(v or 0.0):,.2f}"
    except Exception:
        return "$0.00"


def _occ_type(occ: str) -> str | None:
    try:
        # OCC format ...C... or ...P...
        if 'C' in occ and (occ.rfind('C') > occ.rfind('P')):
            return 'CALL'
        if 'P' in occ and (occ.rfind('P') > occ.rfind('C')):
            return 'PUT'
    except Exception:
        return None
    return None


def sync_broker_positions_to_db(cfg: Dict[str, Any]) -> int:
    """Import existing broker options positions into DB so exit engine can manage them.

    - Only imports long CALL/PUT single legs (debit positions) to match current exit logic.
    - Skips symbols already present as OPEN positions in DB.
    - Returns number of positions imported.
    """
    _, _, _, _, acct = _build_clients(cfg)
    try:
        broker_opts = acct.options_positions()
    except Exception:
        broker_opts = []
    if not broker_opts:
        return 0

    imported = 0
    try:
        for session in get_session():
            # Build set of existing OPEN OCC symbols in DB
            existing_open: set[str] = set()
            try:
                db_positions = session.query(Position).filter(Position.status == PositionStatus.OPEN).all()
                for p in db_positions:
                    for leg in (p.legs_json or []):
                        if isinstance(leg, dict) and leg.get('symbol'):
                            existing_open.add(str(leg['symbol']))
            except Exception:
                pass

            for bp in broker_opts:
                occ = (
                    bp.get('symbol')
                    or bp.get('option_symbol')
                    or (bp.get('option_contract') or {}).get('symbol')
                )
                if not occ:
                    continue
                if occ in existing_open:
                    continue
                # Only import long positions
                qty_raw = bp.get('qty') or bp.get('quantity') or bp.get('position_qty') or bp.get('qty_available')
                try:
                    qty = abs(int(float(qty_raw))) if qty_raw is not None else 0
                except Exception:
                    qty = 0
                if qty <= 0:
                    continue
                side = str(bp.get('side') or bp.get('position_side') or '').lower()
                if side in ('short', 'sell'):
                    continue
                typ = _occ_type(str(occ))
                if typ not in ('CALL', 'PUT'):
                    continue
                # Entry debit (approx) from avg entry price if available
                entry_price = bp.get('avg_entry_price') or bp.get('avg_price') or bp.get('average_price')
                try:
                    entry_debit = float(entry_price) * 100.0 * qty if entry_price is not None else None
                except Exception:
                    entry_debit = None

                structure = PositionStructure.CALL if typ == 'CALL' else PositionStructure.PUT
                pos = Position(
                    ticker=str(bp.get('underlying_symbol') or bp.get('symbol', '')[:6]).strip(),
                    structure=structure,
                    legs_json=[{"symbol": str(occ), "side": "buy", "qty": qty}],
                    opened_at=now_et(),
                    status=PositionStatus.OPEN,
                    entry_debit=entry_debit,
                    notes="imported=true; source=broker",
                )
                session.add(pos)
                imported += 1
            break
    except Exception as e:
        logger.error({"event": "sync_broker_positions_error", "err": str(e)})
        return imported
    return imported


def post_intraday_synopsis(cfg: Dict[str, Any]) -> None:
    """Post a 5-minute synopsis during market hours explaining entry outcomes per symbol."""
    headers, md, oc, orders, acct = _build_clients(cfg)
    disc = cfg.get("discord", {})
    webhook = disc.get("trades_webhook")
    if not webhook:
        return

    # Universe
    uni_cfg = cfg.get("universe", {})
    top_n = int(uni_cfg.get("top_n", 15))
    exclude = uni_cfg.get("exclude", [])
    symbols = md.most_actives(top_n=top_n, exclude=exclude)
    if not symbols:
        static = uni_cfg.get("static", [])
        if static:
            symbols = [s for s in static if s not in exclude][:top_n]

    # Config
    sig_cfg = cfg.get("signals", {})
    sma_long = int(sig_cfg.get("sma_long", 50))
    lookback_bars = int(sig_cfg.get("bars_lookback", max(sma_long + 50, 150)))
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
    open_risk_cap_pct = float(risk_cfg.get("max_open_risk_pct_of_equity", 40))

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

    # DB open risk sum
    try:
        for session in get_session():
            from sqlalchemy import func as _func
            open_risk_now = session.query(_func.coalesce(_func.sum(Position.entry_debit), 0.0)).filter(Position.status == PositionStatus.OPEN).scalar()
            open_risk_sum = float(open_risk_now or 0.0)
            break
    except Exception:
        open_risk_sum = 0.0
    open_risk_cap_abs = (open_risk_cap_pct / 100.0) * max(equity, 0.0)
    slots_left = max(0, max_positions - current_positions)

    import pandas as pd
    lines: list[str] = []
    entered = 0
    eligible = 0

    # Build recent entered IDs to summarize (best-effort)
    recent_ids: list[str] = []

    for sym in symbols:
        reason = None
        try:
            if slots_left <= 0:
                reason = "no slots left"
                lines.append(f"{sym} -> skipped ({reason})")
                continue

            # Trend
            bars = md.daily_bars(sym, lookback=lookback_bars)
            closes = [b.get("close") for b in bars if b.get("close") is not None]
            if len(closes) < sma_long:
                lines.append(f"{sym} -> skipped (insufficient history: {len(closes)} bars)")
                continue
            df = compute_sma20_50(pd.DataFrame({"close": closes}))
            s20 = df["sma20"].iloc[-1]
            s50 = df["sma50"].iloc[-1]
            if pd.isna(s20) or pd.isna(s50):
                lines.append(f"{sym} -> skipped (SMA not available)")
                continue
            direction = "BULL" if float(s20) > float(s50) else ("BEAR" if float(s20) < float(s50) else "NEUTRAL")
            if direction == "NEUTRAL":
                lines.append(f"{sym} -> skipped (NEUTRAL)")
                continue

            # Spot
            spot = md.latest_trade_price(sym)
            if spot is None:
                lines.append(f"{sym} {direction} -> skipped (no spot)")
                continue

            # Contracts & expiry
            cons = oc.contracts(sym, dmin, dmax)
            if not cons:
                cons = oc.contracts(sym, dmin, 28)
            cons = [c for c in cons if c.get("expiration_date")]
            if not cons:
                lines.append(f"{sym} {direction} -> skipped (no contracts)")
                continue
            from datetime import datetime as _dt, date as _date
            def _dte(exp: str) -> int:
                try:
                    d = _dt.fromisoformat(exp).date()
                except Exception:
                    d = _date.fromisoformat(exp)
                return (d - _date.today()).days
            cons.sort(key=lambda c: abs(_dte(c.get("expiration_date", ""))))
            expiry = cons[0]["expiration_date"]
            same_exp = [c for c in cons if c.get("expiration_date") == expiry]
            same_exp.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
            focus = same_exp[:30]
            occ_syms = [c.get("symbol") for c in focus if c.get("symbol")]
            snaps = oc.snapshots(occ_syms)

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
                lines.append(f"{sym} {direction} -> skipped (no ATM options)")
                continue

            # IV regime
            pct = None
            iv_vals = [v for v in (_iv(atm_call["symbol"]), _iv(atm_put["symbol"])) if v is not None]
            atm_iv = sum(iv_vals) / len(iv_vals) if iv_vals else None
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
                lines.append(f"{sym} {direction}+{iv_regime} -> skipped (no strategy)")
                continue

            # Liquidity baseline
            def _liq(occ: str, eff_cap: float | None = None) -> bool:
                s = snaps.get(occ, {}); q = s.get("latest_quote", {})
                oi = s.get("open_interest") or 0; vol = s.get("day", {}).get("volume") or 0
                cap = max_spread_pct if eff_cap is None else eff_cap
                return oi >= min_oi and vol >= min_vol and passes_liquidity(oi, vol, q.get("bid_price"), q.get("ask_price"), cap)
            if not (_liq(atm_call["symbol"]) and _liq(atm_put["symbol"])):
                lines.append(f"{sym} {direction}+{iv_regime} -> skipped (illiquid ATM)")
                continue

            # Singles candidate
            if strat in ("CALL", "PUT"):
                side_list = calls if strat == "CALL" else puts
                from src.selector.strikes import Contract, pick_single_delta_band
                cands: list[Contract] = []
                for c in side_list:
                    s = snaps.get(c["symbol"], {}); g = s.get("greeks", {}); q = s.get("latest_quote", {})
                    cands.append(Contract(symbol=c["symbol"], expiry=expiry, strike=float(c.get("strike_price", 0.0)),
                                          type=("C" if c.get("type") == "call" else "P"), delta=g.get("delta"),
                                          bid=q.get("bid_price"), ask=q.get("ask_price"), mid=_mid(c["symbol"]) or 0.0))
                pick = pick_single_delta_band(cands, d_lo, d_hi)
                if not pick:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (no contract in Δ[{d_lo:.2f},{d_hi:.2f}])")
                    continue
                s = snaps.get(pick.symbol, {}); q = s.get("latest_quote", {})
                oi = s.get("open_interest") or 0; vol = s.get("day", {}).get("volume") or 0
                eff_cap = 0.05 if (pick.mid or 0.0) < 0.80 else max_spread_pct
                if not passes_liquidity(oi, vol, q.get("bid_price"), q.get("ask_price"), eff_cap):
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (illiquid Δ {float(pick.delta or 0):.2f})")
                    continue
                debit = float(pick.mid or 0.0)
                if debit <= 0 or debit < min_premium:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (debit ${debit:.2f} < min ${min_premium:.2f})")
                    continue
                qty = int(max(1, per_max // (debit * 100)))
                if qty < 1 or (debit * 100) < per_min:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (budget too small)")
                    continue
                est_cost = debit * 100.0 * qty
                if (open_risk_sum + est_cost) > open_risk_cap_abs:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (open-risk cap)")
                    continue
                eligible += 1
                lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} K={pick.strike:g} Δ={(pick.delta or 0):.2f} mid=${debit:.2f} qty={qty} est_cost={_format_money(est_cost)}")
            else:
                # Spread
                atm_call_mid = _mid(atm_call["symbol"]) or 0.0
                atm_put_mid = _mid(atm_put["symbol"]) or 0.0
                if atm_call_mid <= 0 or atm_put_mid <= 0:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (bad ATM quotes)")
                    continue
                from src.features.implied_move import implied_move_abs
                im = implied_move_abs(atm_call_mid, atm_put_mid) or width_default
                if direction == "BULL":
                    long_leg = atm_call; target = spot + (im if use_im_cap else width_default); short_pool = calls
                else:
                    long_leg = atm_put; target = spot - (im if use_im_cap else width_default); short_pool = puts
                short_pool.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - target))
                short_leg = short_pool[0] if short_pool else None
                def _liq_leg(c: dict) -> bool:
                    s = snaps.get(c["symbol"], {})
                    q = s.get("latest_quote", {})
                    bid = q.get("bid_price"); ask = q.get("ask_price")
                    if not bid or not ask or bid <= 0 or ask <= 0:
                        return False
                    mid = (bid + ask) / 2.0
                    if mid <= 0:
                        return False
                    if abs(ask - bid) / mid > max_spread_pct:
                        return False
                    oi = s.get("open_interest"); vol = (s.get("day", {}) or {}).get("volume")
                    if oi is not None and oi < min_oi:
                        return False
                    if vol is not None and vol < min_vol:
                        return False
                    return True
                if not short_leg or not _liq_leg(short_leg):
                    alt = spot + (width_default if direction == "BULL" else -width_default)
                    short_pool.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - alt))
                    if short_pool and _liq_leg(short_pool[0]):
                        short_leg = short_pool[0]
                if not short_leg:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (no viable short leg)")
                    continue
                long_mid = _mid(long_leg["symbol"]) or 0.0
                short_mid = _mid(short_leg["symbol"]) or 0.0
                if long_mid <= 0 or short_mid <= 0:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (bad quotes for legs)")
                    continue
                net_debit = max(0.0, long_mid - short_mid)
                if net_debit < min_premium or not (per_min <= net_debit * 100 <= per_max):
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (debit ${net_debit:.2f} outside budget)")
                    continue
                est_cost = net_debit * 100.0
                if (open_risk_sum + est_cost) > open_risk_cap_abs:
                    lines.append(f"{sym} {direction}+{iv_regime} -> skipped (open-risk cap)")
                    continue
                eligible += 1
                try:
                    k_long = float(long_leg.get("strike_price")); k_short = float(short_leg.get("strike_price"))
                    width = abs(k_short - k_long)
                    lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} Klong={k_long:g} Kshort={k_short:g} width=${width:.2f} debit=${net_debit:.2f} qty=1 est_cost={_format_money(est_cost)}")
                except Exception:
                    lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} debit=${net_debit:.2f} qty=1 est_cost={_format_money(est_cost)}")
        except Exception as e:
            lines.append(f"{sym} -> error ({str(e)})")
            continue

    # Compose embed
    slots_after = max(0, max_positions - current_positions)
    desc = f"Open={current_positions} | Slots={slots_left} | Cash={_format_money(cash)} | BP={_format_money(buying_power)} | OpenRisk={_format_money(open_risk_sum)} / Cap={_format_money(open_risk_cap_abs)}"
    joined = "\n".join(lines[:45])  # keep within embed limits
    payload = build_info_embed(
        title="Intraday Synopsis",
        description=desc,
        color=int(disc.get("color_neutral", 15844367)),
        fields=[{"name": "Per-Symbol", "value": f"```\n{joined}\n```", "inline": False}],
    )
    # Use a key with minute bucket to dedupe
    ts_key = int(now_et().timestamp() // 300)
    _send_once(webhook, f"INTRADAY_SYNOPSIS:{today_et()}:{ts_key}", payload)


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
    # Ensure DB is initialized for optional percentile computations and outbox writes
    try:
        init_db()
    except Exception:
        pass
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

    # Market Summary (on start) using SMA20/50 direction per top universe
    try:
        import pandas as pd  # local import to avoid global dep at module import time
        uni_cfg = cfg.get("universe", {})
        top_n = int(uni_cfg.get("top_n", 15))
        exclude = uni_cfg.get("exclude", [])
        symbols = md.most_actives(top_n=top_n, exclude=exclude)
        if not symbols:
            static = uni_cfg.get("static", [])
            if static:
                symbols = [s for s in static if s not in exclude][:top_n]
        bull_syms: List[str] = []
        bear_syms: List[str] = []
        neutral_syms: List[str] = []
        bar_counts_ordered: List[tuple[str, int]] = []
        # Use same expanded lookback as trading to avoid classifying everything as NEUTRAL due to insufficient history
        sig_cfg = cfg.get("signals", {})
        sma_long = int(sig_cfg.get("sma_long", 50))
        lookback_bars = int(sig_cfg.get("bars_lookback", max(sma_long + 50, 150)))
        # Additional config for plan generation
        lookback_iv_days = int(sig_cfg.get("iv_percentile_lookback_days", 90))
        lo_thr = float(sig_cfg.get("iv_low_threshold_pctile", 30))
        hi_thr = float(sig_cfg.get("iv_high_threshold_pctile", 70))
        strat_cfg = cfg.get("strategy", {})
        dmin = int(strat_cfg.get("expiry_min_days", 7))
        dmax = int(strat_cfg.get("expiry_max_days", 21))
        plan_lines: List[str] = []
        # Plan mode: lightweight (no options endpoints) or full (contracts+snapshots) — default lightweight to avoid blocking at startup
        diag_cfg = cfg.get("diagnostics", {})
        plan_mode = str(diag_cfg.get("startup_plan_mode", "lightweight")).lower()
        for sym in symbols:
            try:
                bars = md.daily_bars(sym, lookback=lookback_bars)
                bar_counts_ordered.append((sym, len(bars)))
                closes = [b.get("close") for b in bars if b.get("close") is not None]
                if len(closes) < 50:
                    neutral_syms.append(sym)
                    plan_lines.append(f"{sym} NEUTRAL -> (insufficient history)")
                    continue
                df = compute_sma20_50(pd.DataFrame({"close": closes}))
                sma20 = float(df["sma20"].iloc[-1]) if pd.notnull(df["sma20"].iloc[-1]) else None
                sma50 = float(df["sma50"].iloc[-1]) if pd.notnull(df["sma50"].iloc[-1]) else None
                if sma20 is None or sma50 is None:
                    neutral_syms.append(sym)
                    plan_lines.append(f"{sym} NEUTRAL -> (SMA not available)")
                    continue
                direction = "BULL" if sma20 > sma50 else ("BEAR" if sma20 < sma50 else "NEUTRAL")
                if direction == "NEUTRAL":
                    neutral_syms.append(sym)
                    plan_lines.append(f"{sym} NEUTRAL -> (no trade)")
                    continue
                (bull_syms if direction == "BULL" else bear_syms).append(sym)

                if plan_mode == "full":
                    # Compute IV regime and tentative expiry to pick strategy (may use options endpoints; can be slow)
                    try:
                        spot = md.latest_trade_price(sym)
                        if spot is None:
                            plan_lines.append(f"{sym} {direction}+MID -> (no spot)")
                            continue
                        cons = oc.contracts(sym, dmin, dmax)
                        if not cons:
                            cons = oc.contracts(sym, dmin, 28)
                        cons = [c for c in cons if c.get("expiration_date")]
                        if not cons:
                            plan_lines.append(f"{sym} {direction}+MID -> (no contracts)")
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
                        calls = [c for c in same_exp if c.get("type") == "call"]
                        puts = [c for c in same_exp if c.get("type") == "put"]
                        calls.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
                        puts.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
                        atm_call = calls[0] if calls else None
                        atm_put = puts[0] if puts else None
                        pct = None
                        if atm_call and atm_put:
                            occs = [atm_call.get("symbol"), atm_put.get("symbol")]
                            snaps = oc.snapshots([o for o in occs if o]) if occs else {}
                            def _iv(occ: str) -> float | None:
                                s = snaps.get(occ, {}); g = s.get("greeks", {}); v = g.get("iv"); return float(v) if v is not None else None
                            iv_vals = [v for v in (_iv(atm_call["symbol"]), _iv(atm_put["symbol"])) if v is not None]
                            atm_iv = sum(iv_vals) / len(iv_vals) if iv_vals else None
                            try:
                                for session in get_session():
                                    pct = upsert_iv_summary(session, sym, today_et(), float(atm_call.get("strike_price")) if atm_call else None,
                                                            _dt.fromisoformat(expiry).date() if isinstance(expiry, str) else expiry,
                                                            atm_iv, lookback_iv_days)[1]
                                    break
                            except Exception:
                                pct = None
                        iv_regime = "MID" if pct is None else ("LOW" if pct <= lo_thr else ("HIGH" if pct >= hi_thr else "MID"))
                        strat = choose_strategy(SignalSnapshot(direction=direction, iv_regime=iv_regime))
                        if strat is None:
                            plan_lines.append(f"{sym} {direction}+{iv_regime} -> (no strategy)")
                            continue
                        try:
                            plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} spot=${spot:.2f}")
                        except Exception:
                            plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry}")
                    except Exception:
                        plan_lines.append(f"{sym} {direction}+MID -> (plan error)")
                else:
                    # Lightweight plan: avoid options endpoints; assume MID IV and no expiry fetch
                    iv_regime = "MID"
                    strat = choose_strategy(SignalSnapshot(direction=direction, iv_regime=iv_regime))
                    try:
                        spot = md.latest_trade_price(sym)
                    except Exception:
                        spot = None
                    if strat is None:
                        plan_lines.append(f"{sym} {direction}+{iv_regime} -> (no strategy)")
                    else:
                        if spot is not None:
                            plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} spot=${spot:.2f}")
                        else:
                            plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat}")
            except Exception:
                neutral_syms.append(sym)
                plan_lines.append(f"{sym} NEUTRAL -> (error)")
        logger.info({
            "event": "market_summary_start",
            "bullish": len(bull_syms),
            "bearish": len(bear_syms),
            "neutral": len(neutral_syms),
            "bar_counts": {s: n for s, n in bar_counts_ordered},
        })
        if webhook:
            _send_once(webhook, f"MARKET_SUMMARY:{today_et()}", build_info_embed(
                title="Market Summary (on start)",
                description=f"Bullish={len(bull_syms)} | Bearish={len(bear_syms)} | Neutral={len(neutral_syms)}",
                color=color_neutral,
                fields=[
                    {"name": "Bullish", "value": ", ".join(bull_syms) or "(none)", "inline": False},
                    {"name": "Bearish", "value": ", ".join(bear_syms) or "(none)", "inline": False},
                    {"name": "Neutral", "value": ", ".join(neutral_syms) or "(none)", "inline": False},
                    {"name": "Bars Returned (per symbol)", "value": ", ".join([f"{s}:{n}" for s, n in bar_counts_ordered]) or "(none)", "inline": False},
                ],
            ))
            if plan_lines:
                joined = "\n".join(plan_lines)
                if len(joined) > 1800:
                    joined = joined[:1800] + "\n..."
                _send_once(webhook, f"MARKET_PLAN_START:{today_et()}", {
                    "embeds": [{
                        "title": "Plan (on start)",
                        "description": f"```\n{joined}\n```",
                    }]
                })
    except Exception as e:
        logger.error({"event": "market_summary_error", "err": str(e)})

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

    # (removed duplicate midday Market Summary block; handled earlier with detailed summary)

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
    # Ensure we fetch ample history to compute reliable SMAs
    lookback_bars = int(sig_cfg.get("bars_lookback", max(sma_long + 50, 150)))
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
    plan_lines: List[str] = []  # Per-symbol plan summary to post to Discord
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
            bars = md.daily_bars(sym, lookback=lookback_bars)
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
                # Capture plan line with no-entry rationale
                plan_lines.append(f"{sym} NEUTRAL -> (no trade)")
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
                # Should not happen for non-NEUTRAL, but guard just in case
                plan_lines.append(f"{sym} {direction}+{iv_regime} -> (no strategy)")
                continue

            # Defer adding to plan_lines until after selection/sizing to include details

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
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} (no contract in Δ[{d_lo:.2f},{d_hi:.2f}])")
                    continue
                s = snaps.get(pick.symbol, {}); q = s.get("latest_quote", {})
                oi = s.get("open_interest") or 0; vol = s.get("day", {}).get("volume") or 0
                # Dynamic spread cap: tighten to 5% when premium < $0.80
                eff_cap = 0.05 if (pick.mid or 0.0) < 0.80 else max_spread_pct
                if not _liq(pick.symbol, eff_cap):
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} K={pick.strike:g} (illiquid by spread/OI/Vol)")
                    continue
                debit = pick.mid or 0.0
                if debit <= 0 or debit < min_premium:
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} K={pick.strike:g} (debit ${debit:.2f} < min ${min_premium:.2f})")
                    continue
                qty = int(max(1, per_max // (debit * 100)))
                if qty < 1 or (debit * 100) < per_min:
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} K={pick.strike:g} (budget too small)")
                    continue
                legs = [{"symbol": pick.symbol, "side": "buy", "qty": qty}]
                limit_price = round(debit, 2)
                net_cost = debit * 100.0 * qty
                # Strict open-risk cap gate
                if (open_risk_sum + net_cost) > open_risk_cap_abs:
                    # skip due to open-risk cap
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} K={pick.strike:g} (skipped: open risk cap)")
                    continue
            else:
                if atm_call_mid is None or atm_put_mid is None:
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} (missing ATM quotes)")
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
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} (no viable short leg)")
                    continue
                long_mid = _mid(long_leg["symbol"]) or 0.0
                short_mid = _mid(short_leg["symbol"]) or 0.0
                if long_mid <= 0 or short_mid <= 0:
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} (bad quotes for legs)")
                    continue
                net_debit = max(0.0, long_mid - short_mid)
                if net_debit < min_premium or not (per_min <= net_debit * 100 <= per_max):
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} (debit ${net_debit:.2f} outside budget)")
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
                    plan_lines.append(f"{sym} {direction}+{iv_regime} -> {strat} (skipped: open risk cap)")
                    continue

            # Risk checks vs cash/BP and positions
            available_cash = max(0.0, cash - spent_budget)
            if net_cost > available_cash or net_cost > buying_power:
                continue

            # Submit (and add detailed plan line just before submit)
            try:
                if strat in ("CALL", "PUT"):
                    plan_lines.append(
                        f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} K={pick.strike:g} Δ={(pick.delta or 0):.2f} mid=${limit_price:.2f} qty={qty} est_cost=${net_cost:,.2f}"
                    )
                else:
                    try:
                        k_long = float(long_leg.get("strike_price")); k_short = float(short_leg.get("strike_price"))
                        width = abs(k_short - k_long)
                        plan_lines.append(
                            f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} Klong={k_long:g} Kshort={k_short:g} width=${width:.2f} debit=${limit_price:.2f} qty={qty} est_cost=${net_cost:,.2f}"
                        )
                    except Exception:
                        plan_lines.append(
                            f"{sym} {direction}+{iv_regime} -> {strat} exp={expiry} debit=${limit_price:.2f} qty={qty} est_cost=${net_cost:,.2f}"
                        )
            except Exception:
                pass
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
        # Premarket plan rundown (per symbol)
        if plan_lines:
            # Keep message within Discord limits; truncate if too long
            joined = "\n".join(plan_lines)
            if len(joined) > 1800:
                joined = joined[:1800] + "\n..."
            _send_once(webhook, f"PREMARKET_PLAN:{today_et()}", {
                "embeds": [{
                    "title": "Premarket Plan",
                    "description": f"```\n{joined}\n```",
                }]
            })


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
        # Even if there are no open positions, we may still want to place new ones if enabled
        pass

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

    # Optional: intraday entries (reuses premarket logic, guarded by config and market window)
    try:
        sched = cfg.get("scheduling", {})
        intraday_entries_enabled = bool(sched.get("intraday_entries_enabled", True))
        window = str(sched.get("intraday_entry_window", "09:40-15:30"))
        def _parse_window(w: str) -> tuple[dtime, dtime]:
            try:
                s, e = w.split("-")
                sh, sm = [int(x) for x in s.split(":")]
                eh, em = [int(x) for x in e.split(":")]
                return dtime(hour=sh, minute=sm), dtime(hour=eh, minute=em)
            except Exception:
                return dtime(hour=9, minute=40), dtime(hour=15, minute=30)
        start_w, end_w = _parse_window(window)
        now = now_et()
        if intraday_entries_enabled and _et_between(now, start_w, end_w) and now.weekday() < 5:
            try:
                run_intraday_entries(cfg)
            except Exception as e:
                logger.error({"event": "intraday_entries_error", "err": str(e)})
    except Exception as e:
        logger.error({"event": "intraday_entries_setup_error", "err": str(e)})


def run_intraday_entries(cfg: Dict[str, Any]) -> None:
    """Entry scan during market hours. Mirrors premarket entry logic with lighter messaging."""
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

    # Trading logic (same config as premarket)
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

    placed_count = 0
    spent_budget = 0.0
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
            bars = md.daily_bars(sym, lookback=max(sma_long + 5, 150))
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
                continue

            # Spot
            spot = md.latest_trade_price(sym)
            if spot is None:
                continue
            # Contracts near DTE window
            cons = oc.contracts(sym, dmin, dmax)
            if not cons:
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

            # Liquidity helper: enforce quotes/spread cap; only enforce OI/Vol if present
            def _liq(occ: str, eff_spread_cap: float | None = None) -> bool:
                s = snaps.get(occ, {})
                q = s.get("latest_quote", {})
                bid = q.get("bid_price"); ask = q.get("ask_price")
                cap = max_spread_pct if eff_spread_cap is None else eff_spread_cap
                if not bid or not ask or bid <= 0 or ask <= 0:
                    return False
                mid = (bid + ask) / 2.0
                if mid <= 0:
                    return False
                if abs(ask - bid) / mid > cap:
                    return False
                oi = s.get("open_interest")
                vol = (s.get("day", {}) or {}).get("volume")
                if oi is not None and oi < min_oi:
                    return False
                if vol is not None and vol < min_vol:
                    return False
                return True

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
                eff_cap = 0.05 if (pick.mid or 0.0) < 0.80 else max_spread_pct
                # Use the same liquidity helper logic as premarket (allow missing OI/Vol if spread is tight)
                def _liq_pick(sym_occ: str, cap: float) -> bool:
                    s2 = snaps.get(sym_occ, {})
                    q2 = s2.get("latest_quote", {})
                    bid = q2.get("bid_price"); ask = q2.get("ask_price")
                    if not bid or not ask or bid <= 0 or ask <= 0:
                        return False
                    mid2 = (bid + ask) / 2.0
                    if mid2 <= 0:
                        return False
                    if abs(ask - bid) / mid2 > cap:
                        return False
                    oi2 = s2.get("open_interest"); vol2 = (s2.get("day", {}) or {}).get("volume")
                    if oi2 is not None and oi2 < min_oi:
                        return False
                    if vol2 is not None and vol2 < min_vol:
                        return False
                    return True
                if not _liq_pick(pick.symbol, eff_cap):
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
                if (open_risk_sum + net_cost) > open_risk_cap_abs:
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
                    s2 = snaps.get(c["symbol"], {})
                    q2 = s2.get("latest_quote", {})
                    bid = q2.get("bid_price"); ask = q2.get("ask_price")
                    if not bid or not ask or bid <= 0 or ask <= 0:
                        return False
                    mid2 = (bid + ask) / 2.0
                    if mid2 <= 0:
                        return False
                    if abs(ask - bid) / mid2 > max_spread_pct:
                        return False
                    oi2 = s2.get("open_interest"); vol2 = (s2.get("day", {}) or {}).get("volume")
                    if oi2 is not None and oi2 < min_oi:
                        return False
                    if vol2 is not None and vol2 < min_vol:
                        return False
                    return True
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
                if (open_risk_sum + net_cost) > open_risk_cap_abs:
                    continue

            available_cash = max(0.0, cash - spent_budget)
            if net_cost > available_cash or net_cost > buying_power:
                continue

            try:
                resp = orders.submit_options_order(legs=legs, limit_price=limit_price)
            except Exception as e:
                logger.error({"event": "order_submit_error_intraday", "symbol": sym, "err": str(e)})
                continue
            order_id = resp.get("id", "")
            placed_count += 1
            spent_budget += net_cost
            open_risk_sum += net_cost

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
                logger.error({"event": "persist_position_error_intraday", "symbol": sym, "err": str(e)})

            if webhook:
                cash_after = max(0.0, cash - spent_budget)
                bp_after = max(0.0, buying_power - net_cost)
                slots_left = max(0, max_positions - (current_positions + placed_count))
                _send_once(webhook, f"NEW_INTRADAY:{order_id}", build_trade_new_embed(
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

            try:
                od = orders.wait_for_status(order_id, ["filled", "partially_filled"], timeout_sec=5.0)
                if od and webhook:
                    fill_price = od.get("filled_avg_price") or od.get("limit_price")
                    _send_once(webhook, f"FILLED_INTRADAY:{order_id}", build_filled_embed(
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
            logger.error({"event": "intraday_entries_symbol_error", "symbol": sym, "err": str(e)})
            continue

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
                {"name": "Greeks", "value": f"Delta {greeks['delta']:.2f} | Theta {greeks['theta']:.2f} | Vega {greeks['vega']:.2f}", "inline": False},
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
        # Fallback: if no DB open positions, try broker options positions to populate a basic snapshot
        if not open_positions and eod:
            try:
                broker_opts = acct.options_positions()
                occs2 = [p.get("symbol") for p in broker_opts if isinstance(p, dict) and p.get("symbol")]
                snaps2 = oc.snapshots(occs2) if occs2 else {}
                unreal2 = 0.0
                open_risk2 = 0.0
                greeks2 = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
                for bp in broker_opts:
                    occ = bp.get("symbol"); qty = float(bp.get("qty") or bp.get("quantity") or bp.get("position_qty") or 0)
                    side = str(bp.get("side") or bp.get("position_side") or "").lower()
                    sgn = 1.0
                    if side in ("short", "sell") or qty < 0:
                        sgn = -1.0
                    qty_abs = abs(int(qty)) if qty else 0
                    snap = snaps2.get(occ, {})
                    q = snap.get("latest_quote", {})
                    bpv = q.get("bid_price"); apv = q.get("ask_price")
                    mid = ((bpv or 0) + (apv or 0)) / 2.0 if bpv and apv else None
                    entry_price = bp.get("avg_entry_price") or bp.get("avg_price") or bp.get("average_price")
                    try:
                        entry = float(entry_price) * 100.0 * qty_abs if entry_price is not None else 0.0
                    except Exception:
                        entry = 0.0
                    curr = (float(mid) * 100.0 * qty_abs) if (mid and qty_abs) else 0.0
                    open_risk2 += max(0.0, entry)
                    unreal2 += (sgn * (curr - entry))
                    g = snap.get("greeks", {}) if isinstance(snap, dict) else {}
                    greeks2["delta"] += sgn * float(g.get("delta") or 0.0) * qty_abs
                    greeks2["theta"] += sgn * float(g.get("theta") or 0.0) * qty_abs
                    greeks2["vega"] += sgn * float(g.get("vega") or 0.0) * qty_abs
                total_pl2 = unreal2  # realized unknown from broker snapshot
                fields2 = [
                    {"name": "Cash", "value": f"${cash:,.2f}", "inline": True},
                    {"name": "Equity", "value": f"${equity:,.2f}", "inline": True},
                    {"name": "Unrealized (broker)", "value": f"${unreal2:,.2f}", "inline": True},
                    {"name": "Open Risk (broker)", "value": f"${open_risk2:,.2f}", "inline": True},
                    {"name": "Greeks (broker)", "value": f"Delta {greeks2['delta']:.2f} | Theta {greeks2['theta']:.2f} | Vega {greeks2['vega']:.2f}", "inline": False},
                ]
                _send_once(eod, f"EOD_BROKER:{today}", {
                    "embeds": [{
                        "title": "EOD Summary (Broker Fallback)",
                        "description": "Broker options positions snapshot (no DB positions found)",
                        "fields": fields2,
                    }]
                })
            except Exception as _e:
                logger.error({"event": "eod_broker_fallback_error", "err": str(_e)})

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
    # Prefer minutes if provided; fallback to seconds; default 5 minutes
    _syn_minutes = sched.get("intraday_synopsis_every_minutes")
    if _syn_minutes is not None:
        try:
            synopsis_every = int(float(_syn_minutes) * 60)
        except Exception:
            synopsis_every = 300
    else:
        synopsis_every = int(sched.get("intraday_synopsis_every_seconds", 300))

    pre_start = dtime(hour=pre_mh[1], minute=pre_mh[0])
    market_open = dtime(hour=9, minute=30)
    eod_time = dtime(hour=eod_mh[1], minute=eod_mh[0])
    market_close_guard = dtime(hour=18, minute=0)

    last_intraday_ts: float | None = None
    last_synopsis_ts: float | None = None
    premarket_done_for: str | None = None
    eod_done_for: str | None = None

    logger.info({
        "event": "daemon_start",
        "premarket_start": str(pre_start),
        "market_open": str(market_open),
        "eod_time": str(eod_time),
        "intraday_every": intraday_every,
    })

    # Track config file mtime for hot-reload
    cfg_path = Path("config.yaml")
    try:
        last_cfg_mtime = cfg_path.stat().st_mtime if cfg_path.exists() else None
    except Exception:
        last_cfg_mtime = None

    try:
        # Optional: import existing broker options positions once at start
        sync_cfg = cfg.get("sync", {})
        if bool(sync_cfg.get("broker_positions_on_start", True)):
            try:
                imported = sync_broker_positions_to_db(cfg)
                if imported > 0:
                    logger.info({"event": "sync_broker_positions", "imported": imported})
            except Exception as e:
                logger.error({"event": "sync_broker_positions_error", "err": str(e)})

        while True:
            # Hot-reload config.yaml if modified; apply timezone if changed
            try:
                mtime = cfg_path.stat().st_mtime if cfg_path.exists() else None
                if mtime and last_cfg_mtime and mtime > last_cfg_mtime:
                    new_cfg = load_config(cfg_path)
                    # Update timezone immediately if changed
                    try:
                        tz_name = new_cfg.get("timezone", cfg.get("timezone", "America/New_York"))
                        set_local_tz(tz_name)
                    except Exception:
                        pass
                    cfg = new_cfg  # use updated config for subsequent phases
                    sched = cfg.get("scheduling", {})
                    intraday_every = int(sched.get("intraday_every_seconds", intraday_every))
                    _syn_minutes = sched.get("intraday_synopsis_every_minutes")
                    if _syn_minutes is not None:
                        try:
                            synopsis_every = int(float(_syn_minutes) * 60)
                        except Exception:
                            pass
                    else:
                        synopsis_every = int(sched.get("intraday_synopsis_every_seconds", synopsis_every))
                    last_cfg_mtime = mtime
                    logger.info({"event": "config_reloaded"})
                elif mtime and last_cfg_mtime is None:
                    last_cfg_mtime = mtime
            except Exception as e:
                logger.error({"event": "config_reload_error", "err": str(e)})

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
                # Periodic intraday synopsis (non-blocking if it fails)
                if last_synopsis_ts is None or (ts - last_synopsis_ts) >= synopsis_every:
                    try:
                        post_intraday_synopsis(cfg)
                    except Exception as e:
                        logger.error({"event": "intraday_synopsis_error", "err": str(e)})
                    finally:
                        last_synopsis_ts = ts
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
