from __future__ import annotations

from datetime import date
from typing import Any, Dict, List


class OptionsClient:
    def __init__(self, data_url: str, headers: Dict[str, str]):
        self.data_url = data_url
        self.headers = headers

    def option_chain(self, symbol: str, dte_min: int, dte_max: int) -> List[Dict[str, Any]]:
        # TODO: Implement chain fetch with greeks/IV/OI/volume
        return []

    def snapshot(self, symbol: str) -> Dict[str, Any]:
        # TODO: One-shot snapshot for ATM legs
        return {}

