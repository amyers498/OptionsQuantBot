from __future__ import annotations

import os
from typing import Dict


def get_api_keys() -> Dict[str, str]:
    key = os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

