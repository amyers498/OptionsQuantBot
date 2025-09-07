from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from src.utils.logging import get_logger


logger = get_logger(__name__)


def _build_headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}


def send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    try:
        # Placeholder: no retries here; wrap with tenacity in production
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=_build_headers(), content=json.dumps(payload))
            if resp.status_code // 100 == 2:
                return True
            logger.error({"event": "discord_error", "status": resp.status_code, "body": resp.text})
            return False
    except Exception as e:
        logger.error({"event": "discord_exception", "err": str(e)})
        return False


def build_trade_new_embed(**kwargs: Any) -> Dict[str, Any]:
    # Minimal payload shape; extend with templates later
    return {
        "content": None,
        "embeds": [
            {
                "title": kwargs.get("title", "[NEW] Trade Submitted"),
                "description": kwargs.get("description", ""),
                "color": kwargs.get("color", 15844367),
                "fields": kwargs.get("fields", []),
            }
        ],
    }

