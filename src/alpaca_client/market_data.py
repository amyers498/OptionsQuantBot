from __future__ import annotations

from typing import Any, Dict, List

# Placeholder module for Alpaca market data interactions


class MarketDataClient:
    def __init__(self, base_url: str, data_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.data_url = data_url
        self.headers = headers

    def most_actives(self, top_n: int = 15, exclude: List[str] | None = None) -> List[str]:
        # TODO: Implement via Alpaca data API (most actives/movers)
        return []

    def daily_bars(self, symbol: str, lookback: int = 100) -> List[Dict[str, Any]]:
        # TODO: Fetch daily bars for SMA computations
        return []

