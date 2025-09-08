from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx


class OptionsClient:
    def __init__(self, data_url: str, headers: Dict[str, str], *, trading_url: str | None = None, options_feed: str = "indicative", contracts_source: str = "auto"):
        self.data_url = data_url.rstrip("/")
        self.trading_url = (trading_url or "").rstrip("/") if trading_url else None
        self.headers = headers
        self.options_feed = options_feed
        self.contracts_source = contracts_source  # auto | trading | data

    def _get(self, base: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = base.rstrip("/")
        url = path if path.startswith("http") else f"{base}{path}"
        with httpx.Client(timeout=20.0) as client:
            r = client.get(url, headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    def _contracts_page(self, base: str, path: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        data = self._get(base, path, params=params)
        contracts = data.get("contracts") or data.get("option_contracts") or []
        next_token = data.get("page_token") or data.get("next_page_token")
        return contracts, next_token

    def contracts(self, symbol: str, dte_min: int, dte_max: int, *, limit_per_page: int = 1000, max_pages: int = 10, source: str | None = None) -> List[Dict[str, Any]]:
        today = date.today()
        gte = (today + timedelta(days=dte_min)).isoformat()
        lte = (today + timedelta(days=dte_max)).isoformat()
        base_params = {
            "underlying_symbols": symbol,
            "status": "active",
            "expiration_date_gte": gte,
            "expiration_date_lte": lte,
            "limit": limit_per_page,
        }

        # Choose base order based on configured source
        use = (source or self.contracts_source or "auto").lower()
        bases: list[tuple[str, str]] = []
        if use == "trading":
            if self.trading_url:
                bases.append((self.trading_url, "/v2/options/contracts"))
        elif use == "data":
            bases.extend(
                [
                    (self.data_url, "/v1beta1/options/contracts"),
                    (self.data_url, "/v2/options/contracts"),
                ]
            )
        else:  # auto: try trading first (works in your environment), then data endpoints
            if self.trading_url:
                bases.append((self.trading_url, "/v2/options/contracts"))
            bases.extend(
                [
                    (self.data_url, "/v1beta1/options/contracts"),
                    (self.data_url, "/v2/options/contracts"),
                ]
            )

        # Try with filters first; if empty across bases, relax filters
        for relax in (False, True):
            params = dict(base_params)
            if relax:
                params.pop("expiration_date_gte", None)
                params.pop("expiration_date_lte", None)
                params["status"] = "all"
            for base, path in bases:
                try:
                    page_token: Optional[str] = None
                    all_cons: List[Dict[str, Any]] = []
                    pages = 0
                    while pages < max_pages:
                        local = dict(params)
                        if page_token:
                            local["page_token"] = page_token
                        cons, page_token = self._contracts_page(base, path, local)
                        all_cons.extend(cons)
                        pages += 1
                        if not page_token:
                            break
                    if all_cons:
                        # Normalize: if endpoint returns strings, wrap into dicts with 'symbol'
                        norm: List[Dict[str, Any]] = []
                        for c in all_cons:
                            if isinstance(c, str):
                                norm.append({"symbol": c})
                            else:
                                norm.append(c)
                        return norm
                except httpx.HTTPStatusError:
                    continue
                except Exception:
                    continue
        return []

    def contracts_probe(self, symbol: str, dte_min: int, dte_max: int) -> dict:
        """Return counts and HTTP status by endpoint/relaxation to aid debugging."""
        today = date.today()
        gte = (today + timedelta(days=dte_min)).isoformat()
        lte = (today + timedelta(days=dte_max)).isoformat()
        base_params = {
            "underlying_symbols": symbol,
            "status": "active",
            "expiration_date_gte": gte,
            "expiration_date_lte": lte,
            "limit": 1000,
        }
        bases: list[tuple[str, str, str]] = [
            (self.data_url, "/v1beta1/options/contracts", "data_v1beta1"),
            (self.data_url, "/v2/options/contracts", "data_v2"),
        ]
        if self.trading_url:
            bases.append((self.trading_url, "/v2/options/contracts", "trading_v2"))
        out: dict[str, dict] = {}
        for relax in (False, True):
            params = dict(base_params)
            key_suffix = ""
            if relax:
                params.pop("expiration_date_gte", None)
                params.pop("expiration_date_lte", None)
                params["status"] = "all"
                key_suffix = "_relaxed"
            for base, path, name in bases:
                key = f"{name}{key_suffix}"
                url = f"{base.rstrip('/')}{path}"
                try:
                    with httpx.Client(timeout=15.0) as client:
                        r = client.get(url, headers=self.headers, params=params)
                        status = r.status_code
                        if r.status_code // 100 == 2:
                            data = r.json()
                            cons = data.get("contracts") or data.get("option_contracts") or []
                            out[key] = {"count": len(cons), "status": status}
                        else:
                            out[key] = {"count": -1, "status": status, "body": r.text[:200]}
                except Exception as e:
                    out[key] = {"count": -1, "status": None, "error": str(e)}
        return out

    def snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}
        # Chunk requests if needed (limit URL length)
        out: Dict[str, Dict[str, Any]] = {}
        step = 100
        for i in range(0, len(symbols), step):
            chunk = symbols[i : i + step]
            data = self._get(
                self.data_url,
                "/v1beta1/options/snapshots",
                params={"symbols": ",".join(chunk), "feed": self.options_feed},
            )
            snaps = data.get("snapshots") or []
            if isinstance(snaps, dict):
                for sym, snap in snaps.items():
                    if not sym or not isinstance(snap, dict):
                        continue
                    out[sym] = snap
            else:
                for s in snaps:
                    if not isinstance(s, dict):
                        continue
                    sym = s.get("symbol")
                    if not sym:
                        continue
                    out[sym] = s
        return out
