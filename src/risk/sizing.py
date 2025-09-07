from __future__ import annotations

from math import floor


def max_contracts_for_budget(net_debit_per_contract: float, per_trade_min: float, per_trade_max: float) -> int:
    if net_debit_per_contract <= 0:
        return 0
    max_contracts = floor(per_trade_max / (net_debit_per_contract * 100))
    min_contracts = 1 if (net_debit_per_contract * 100) >= per_trade_min else 0
    return max(0, max(max_contracts, min_contracts))

