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

    def options_positions(self) -> List[Dict[str, Any]]:
        """Return open options positions if the endpoint is available.

        Falls back to empty list on 404 or network errors.
        """
        url = f"{self.base_url}/v2/options/positions"
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url, headers=self.headers)
                if r.status_code == 404:
                    return []
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and data.get("positions"):
                    return data.get("positions") or []
                return data if isinstance(data, list) else []
        except Exception:
            return []
