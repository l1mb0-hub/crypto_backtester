"""Стратегии на тиках.

Каждая стратегия получает DataFrame тиков и возвращает целевую позицию (-1..1)
на каждом тике. Окна задаются временем ("30s", "5min"), а не числом свечей:
тики идут неравномерно. Сигнал считается по mid, если он есть (см. backtest.add_quotes),
иначе по цене сделки.
"""
import numpy as np
import pandas as pd


def _price(ticks):
    return ticks["mid"] if "mid" in ticks else ticks["price"]


def _after_warmup(values, ticks, window):
    """NaN до тех пор, пока с начала данных не прошло полное окно."""
    values = np.asarray(values, dtype=float).copy()
    values[ticks.index < ticks.index[0] + pd.Timedelta(window)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _hold(position, index):
    """NaN означает "держать предыдущую позицию"."""
    return pd.Series(position, index=index).ffill().fillna(0.0)


def mean_reversion(ticks, window="5min", entry_z=2.0, exit_z=0.5):
    price = _price(ticks)
    roll = price.rolling(window)
    z = (price.to_numpy() - roll.mean().to_numpy()) / roll.std().to_numpy()
    z = _after_warmup(z, ticks, window)

    position = np.full(len(ticks), np.nan)
    position[z > entry_z] = -1
    position[z < -entry_z] = 1
    position[np.abs(z) < exit_z] = 0
    return _hold(position, ticks.index)


def momentum(ticks, fast="1min", slow="15min", band_bps=2.0):
    """Пересечение скользящих средних. На тиках разница MA постоянно дребезжит около нуля,
    поэтому позиция меняется только при выходе за полосу band_bps."""
    price = _price(ticks)
    fast_ma = price.rolling(fast).mean().to_numpy()
    slow_ma = price.rolling(slow).mean().to_numpy()
    diff_bps = _after_warmup((fast_ma / slow_ma - 1) * 10_000, ticks, slow)

    position = np.full(len(ticks), np.nan)
    position[diff_bps > band_bps] = 1
    position[diff_bps < -band_bps] = -1
    return _hold(position, ticks.index)


def flow_imbalance(ticks, window="1min", entry=0.4, exit=0.1):
    """Дисбаланс потока сделок: (объём агрессивных покупок - продаж) / весь объём за окно.
    Доступен только на тиках, т.к. нужен признак агрессора каждой сделки."""
    signed = ticks["qty"] * ticks["side"]
    imbalance = signed.rolling(window).sum().to_numpy() / ticks["qty"].rolling(window).sum().to_numpy()
    imbalance = _after_warmup(imbalance, ticks, window)

    position = np.full(len(ticks), np.nan)
    position[imbalance > entry] = 1
    position[imbalance < -entry] = -1
    position[np.abs(imbalance) < exit] = 0
    return _hold(position, ticks.index)


STRATEGIES = {
    "mean_reversion": mean_reversion,
    "momentum": momentum,
    "flow_imbalance": flow_imbalance,
}
