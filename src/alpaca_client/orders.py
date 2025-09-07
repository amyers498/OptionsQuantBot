from __future__ import annotations

from typing import Any, Dict, List


class OrdersClient:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.headers = headers

    def submit_order(self, legs: List[Dict[str, Any]], client_order_id: str | None = None) -> Dict[str, Any]:
        # TODO: Submit single/multi-leg paper order at limit
        return {"id": "stub-order-id", "status": "accepted"}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        # TODO: Fetch order status
        return {"id": order_id, "status": "filled"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        # TODO: Cancel order
        return {"id": order_id, "status": "canceled"}

