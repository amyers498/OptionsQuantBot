from __future__ import annotations

from typing import Any, Dict, List


class AccountClient:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.headers = headers

    def account(self) -> Dict[str, Any]:
        # TODO: Return cash, equity, etc.
        return {"cash": 5000.0, "equity": 5000.0}

    def positions(self) -> List[Dict[str, Any]]:
        # TODO: Return open positions
        return []

