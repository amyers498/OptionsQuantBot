from __future__ import annotations

from typing import Any, Dict, List

import httpx


class AccountClient:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url.rstrip("/")
        self.headers = headers

    def account(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v2/account"
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, headers=self.headers)
            r.raise_for_status()
            return r.json()

    def positions(self) -> List[Dict[str, Any]]:
        # Equity positions; options positions may have a dedicated endpoint in future
        url = f"{self.base_url}/v2/positions"
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, headers=self.headers)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return r.json()
