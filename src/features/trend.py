from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_sma20_50(df: pd.DataFrame) -> pd.DataFrame:
    # Expects columns: ['date','close']
    df = df.copy()
    df["sma20"] = sma(df["close"], 20)
    df["sma50"] = sma(df["close"], 50)
    return df

