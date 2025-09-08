from __future__ import annotations

import asyncio
from typing import Dict, Any


async def _probe_ws(feed: str, key: str, secret: str) -> tuple[bool, str]:
    try:
        import websockets
        import msgpack
    except Exception as e:
        return False, f"missing deps: {e}"

    url = f"wss://stream.data.alpaca.markets/v1beta1/{feed}"
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            auth_msg = {"action": "auth", "key": key, "secret": secret}
            await ws.send(msgpack.packb(auth_msg))

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                return False, "timeout waiting auth response"

            try:
                msg = msgpack.unpackb(raw, raw=False)
            except Exception:
                return False, "invalid msgpack response"

            # Expect something like {"T":"success","msg":"authenticated"} or an error
            if isinstance(msg, list) and msg:
                msg = msg[0]
            if isinstance(msg, dict):
                if msg.get("T") in ("success", "authenticated") or msg.get("msg") == "authenticated":
                    return True, "authenticated"
                if msg.get("T") == "error":
                    return False, f"error code={msg.get('code')} msg={msg.get('msg')}"
            return False, f"unexpected {msg}"
    except Exception as e:
        return False, str(e)


def probe_options_stream(feed: str, headers: Dict[str, str]) -> tuple[bool, str]:
    """Attempt to authenticate to the options websocket (msgpack)."""
    key = headers.get("APCA-API-KEY-ID", "")
    secret = headers.get("APCA-API-SECRET-KEY", "")
    if not key or not secret:
        return False, "missing api keys"
    try:
        return asyncio.run(_probe_ws(feed, key, secret))
    except RuntimeError:
        # In case this is called from within an event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_probe_ws(feed, key, secret))

