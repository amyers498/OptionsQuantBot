from __future__ import annotations

from datetime import datetime
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
        data = self._get(
            f"/v2/stocks/{symbol}/bars",
            params={"timeframe": "1Day", "limit": lookback, "feed": self.stocks_feed},
        )
        bars = data.get("bars") or []
        # Normalize
        return [
            {
                "t": b.get("t"),
                "close": b.get("c"),
                "volume": b.get("v"),
            }
            for b in bars
        ]

    def latest_trade_price(self, symbol: str) -> Optional[float]:
        try:
            data = self._get(f"/v2/stocks/{symbol}/trades/latest", params={"feed": self.stocks_feed})
            trade = data.get("trade", {})
            return float(trade.get("p")) if trade and trade.get("p") is not None else None
        except Exception:
            return None
