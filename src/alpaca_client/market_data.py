from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx


class MarketDataClient:
    def __init__(self, base_url: str, data_url: str, headers: Dict[str, str], *, stocks_feed: str = "iex"):
        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self.headers = headers
        self.stocks_feed = stocks_feed

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = path if path.startswith("http") else f"{self.data_url}{path}"
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    def most_actives(self, top_n: int = 15, exclude: List[str] | None = None) -> List[str]:
        exclude = exclude or []
        # Try screener endpoint; fallback to empty list on failure
        try:
            data = self._get("/v1beta1/screener/stocks/most-actives", params={"top": top_n})
            items = data.get("most_actives") or data.get("assets") or []
            syms = [it.get("symbol") for it in items if it.get("symbol")]
            return [s for s in syms if s not in exclude]
        except Exception:
            return []

    def daily_bars(self, symbol: str, lookback: int = 100) -> List[Dict[str, Any]]:
        # v2 per-symbol bars endpoint
        # Provide explicit start/end to avoid API defaults that can yield a single bar
        end = datetime.now(timezone.utc)
        # Request a wide date range to comfortably cover `lookback` bars even with gaps/holidays
        start = end - timedelta(days=max(lookback * 3, 365))
        start_iso = start.isoformat().replace("+00:00", "Z")
        end_iso = end.isoformat().replace("+00:00", "Z")
        data = self._get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe": "1Day",
                "limit": lookback,
                "start": start_iso,
                "end": end_iso,
                "adjustment": "raw",
                "feed": self.stocks_feed,
            },
        )
        bars = data.get("bars") or []
        # Normalize then sort chronologically (oldest -> newest) to ensure SMA windows align correctly
        rows = [
            {
                "t": b.get("t"),
                "close": b.get("c"),
                "volume": b.get("v"),
            }
            for b in bars
            if b is not None
        ]
        # Robust sort: parse ISO strings or use numeric epoch if provided
        def _ts_key(v):
            t = v.get("t")
            try:
                if isinstance(t, (int, float)):
                    return float(t)
                if isinstance(t, str):
                    # Handle 'Z' suffix
                    s = t.replace("Z", "+00:00")
                    return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0
            return 0.0
        try:
            rows.sort(key=_ts_key)
        except Exception:
            # Best-effort: if timestamp shape is unexpected, leave order as-is
            pass
        return rows

    def latest_trade_price(self, symbol: str) -> Optional[float]:
        try:
            data = self._get(f"/v2/stocks/{symbol}/trades/latest", params={"feed": self.stocks_feed})
            trade = data.get("trade", {})
            return float(trade.get("p")) if trade and trade.get("p") is not None else None
        except Exception:
            return None
