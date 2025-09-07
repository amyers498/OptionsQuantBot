from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalSnapshot:
    direction: str  # BULL, BEAR, NEUTRAL
    iv_regime: str  # LOW, HIGH, MID


def choose_strategy(sig: SignalSnapshot) -> str | None:
    if sig.direction == "NEUTRAL":
        return None
    if sig.direction == "BULL" and sig.iv_regime == "LOW":
        return "CALL"
    if sig.direction == "BULL" and sig.iv_regime in ("HIGH", "MID"):
        return "BULL_CALL_SPREAD"
    if sig.direction == "BEAR" and sig.iv_regime == "LOW":
        return "PUT"
    if sig.direction == "BEAR" and sig.iv_regime in ("HIGH", "MID"):
        return "BEAR_PUT_SPREAD"
    return None

