import numpy as np
import pandas as pd


def mean_reversion(df, window=60, entry_z=2.0):
    close = df["close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std

    position = pd.Series(np.nan, index=df.index)
    position[z > entry_z] = -1
    position[z < -entry_z] = 1
    position[z.abs() < 0.5] = 0
    return position.ffill().fillna(0)


def momentum(df, fast=20, slow=100):
    close = df["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return np.sign(fast_ma - slow_ma).fillna(0)


STRATEGIES = {
    "mean_reversion": mean_reversion,
    "momentum": momentum,
}
