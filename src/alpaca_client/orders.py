from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class OrdersClient:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url.rstrip("/")
        self.headers = headers

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=20.0) as client:
            r = client.post(url, headers=self.headers, json=json)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, headers=self.headers)
            r.raise_for_status()
            return r.json()

    def submit_options_order(
        self,
        legs: List[Dict[str, Any]],
        limit_price: float,
        client_order_id: Optional[str] = None,
        time_in_force: str = "day",
        parent_qty: int | None = None,
    ) -> Dict[str, Any]:
        """Submits an options order using Trading API /v2/orders.

        - Single-leg: posts a simple limit order with asset_class=option
        - Multi-leg: posts an order_class=mleg with legs including ratio_qty and position_intent
        """
        if len(legs) == 1:
            leg = legs[0]
            side = leg.get("side", "buy")
            qty = int(leg.get("qty", 1))
            payload = {
                "symbol": leg["symbol"],
                "asset_class": "option",
                "qty": str(qty),
                "side": side,
                "type": "limit",
                "time_in_force": time_in_force,
                "limit_price": round(float(limit_price), 2),
                "position_intent": leg.get("position_intent") or ("buy_to_open" if side == "buy" else "sell_to_open"),
            }
            if client_order_id:
                payload["client_order_id"] = client_order_id
            return self._post("/v2/orders", payload)

        # Multi-leg
        qty = int(parent_qty or legs[0].get("qty", 1))
        def _leg_payload(leg: Dict[str, Any]) -> Dict[str, Any]:
            side = leg.get("side", "buy")
            ratio = int(leg.get("qty", 1))
            return {
                "symbol": leg["symbol"],
                "ratio_qty": str(ratio),
                "side": side,
                "position_intent": leg.get("position_intent") or ("buy_to_open" if side == "buy" else "sell_to_open"),
            }

        payload = {
            "order_class": "mleg",
            "qty": str(qty),
            "type": "limit",
            "limit_price": round(float(limit_price), 2),
            "time_in_force": time_in_force,
            "legs": [_leg_payload(l) for l in legs],
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return self._post("/v2/orders", payload)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._get(f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v2/orders/{order_id}"
        with httpx.Client(timeout=15.0) as client:
            r = client.delete(url, headers=self.headers)
            r.raise_for_status()
            return r.json() if r.text else {"id": order_id, "status": "canceled"}

    def wait_for_status(self, order_id: str, target_statuses: list[str], timeout_sec: float = 10.0, poll_interval: float = 1.0) -> Dict[str, Any] | None:
        import time as _t
        end = _t.time() + timeout_sec
        while _t.time() < end:
            try:
                o = self.get_order(order_id)
                if str(o.get("status", "")).lower() in [s.lower() for s in target_statuses]:
                    return o
            except Exception:
                pass
            _t.sleep(poll_interval)
        return None
