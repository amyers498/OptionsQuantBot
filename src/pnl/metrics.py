from __future__ import annotations


def unrealized_pl(current_value: float, cost_basis: float) -> float:
    return current_value - cost_basis


def total_pl(realized: float, unreal: float) -> float:
    return realized + unreal


def percent_to_max(current_value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return current_value / max_value


def ror(current_value: float, entry_debit: float) -> float:
    if entry_debit <= 0:
        return 0.0
    return (current_value - entry_debit) / entry_debit

