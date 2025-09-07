from src.features.liquidity import passes_liquidity


def test_passes_liquidity():
    assert passes_liquidity(oi=1000, vol=200, bid=1.0, ask=1.1, max_spread_pct=0.2)

