from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

import httpx

from src.utils.logging import get_logger


logger = get_logger(__name__)


def _build_headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}


def _clip(s: Optional[str], max_len: int) -> str:
    try:
        s = "" if s is None else str(s)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s
    except Exception:
        return ""


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload or {})
    embeds: List[Dict[str, Any]] = p.get("embeds") or []
    if not isinstance(embeds, list):
        p["embeds"] = []
        return p
    out_embeds: List[Dict[str, Any]] = []
    for emb in embeds[:10]:  # Discord allows up to 10 embeds
        if not isinstance(emb, dict):
            continue
        e = dict(emb)
        if "title" in e:
            e["title"] = _clip(e.get("title"), 256)
        if "description" in e:
            e["description"] = _clip(e.get("description"), 4000)
        fields = e.get("fields") or []
        if isinstance(fields, list):
            clean_fields = []
            for f in fields[:25]:  # max 25 fields
                if not isinstance(f, dict):
                    continue
                name = _clip(f.get("name"), 256)
                value = _clip(f.get("value"), 1024)
                clean_fields.append({"name": name or "(none)", "value": value or "(none)", "inline": bool(f.get("inline", False))})
            e["fields"] = clean_fields
        out_embeds.append(e)
    p["embeds"] = out_embeds
    return p


def send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    try:
        # Placeholder: no retries here; wrap with tenacity in production
        with httpx.Client(timeout=10.0) as client:
            safe = _sanitize_payload(payload)
            resp = client.post(url, headers=_build_headers(), content=json.dumps(safe))
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


def build_filled_embed(title: str, *, fields: list[dict], color: int | None = None) -> Dict[str, Any]:
    return {
        "content": None,
        "embeds": [
            {
                "title": title,
                "description": "[FILLED]",
                "color": color or 3066993,
                "fields": fields,
            }
        ],
    }


def build_closed_embed(title: str, *, description: str, fields: list[dict], color: int | None = None) -> Dict[str, Any]:
    return {
        "content": None,
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color or 15158332,
                "fields": fields,
            }
        ],
    }


def build_info_embed(title: str, description: str = "", *, color: int | None = None, fields: list[dict] | None = None) -> Dict[str, Any]:
    return {
        "content": None,
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color or 15844367,
                "fields": fields or [],
            }
        ],
    }
