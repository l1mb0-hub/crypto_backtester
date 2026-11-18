"""Тиковый бэктест.

Модель исполнения:
- сигнал на тике i исполняется на первом тике с временем >= t_i + latency (и не раньше i+1);
- execution="quote" (по умолчанию): позиция оценивается по mid, при каждом изменении
  позиции платится половина спреда. Bid/ask восстанавливаются из сделок: покупка агрессора
  проходит по ask, продажа по bid;
- execution="trade": исполнение и оценка по цене следующей сделки (как раньше на свечах).
  Спред здесь не оплачивается, а цена входа случайно оказывается то bid, то ask, поэтому
  для частых сделок результат завышен;
- комиссия fee_bps и дополнительное проскальзывание slippage_bps с каждого изменения позиции.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class Result:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    metrics: dict


def add_quotes(ticks, ttl="500ms", window=2000):
    """Добавляет оценки half_spread и mid.

    Спред = последняя цена покупки агрессора - последняя цена продажи агрессора, если обе
    свежее ttl; сглаживается скользящей медианой по `window` тикам.
    """
    price = ticks["price"].to_numpy()
    buy = ticks["side"].to_numpy() > 0
    t = ticks.index.as_unit("ns").asi8

    ask = pd.Series(np.where(buy, price, np.nan)).ffill().to_numpy()
    bid = pd.Series(np.where(~buy, price, np.nan)).ffill().to_numpy()
    t_ask = pd.Series(np.where(buy, t, np.nan)).ffill().to_numpy()
    t_bid = pd.Series(np.where(~buy, t, np.nan)).ffill().to_numpy()

    ttl_ns = pd.Timedelta(ttl).value
    raw = ask - bid
    valid = (t - t_ask <= ttl_ns) & (t - t_bid <= ttl_ns) & (raw > 0)
    spread = (pd.Series(np.where(valid, raw, np.nan))
              .rolling(window, min_periods=1).median()
              .ffill().fillna(0.0).to_numpy())

    out = ticks.copy()
    out["half_spread"] = spread / 2
    out["mid"] = np.where(buy, price - spread / 2, price + spread / 2)
    return out


def run_backtest(ticks, target_position, fee_bps=4.0, latency_ms=0.0,
                 execution="quote", slippage_bps=0.0):
    if execution not in ("quote", "trade"):
        raise ValueError("execution должен быть 'quote' или 'trade'")
    if execution == "quote" and "mid" not in ticks:
        ticks = add_quotes(ticks)

    n = len(ticks)
    t = ticks.index.as_unit("ns").asi8
    target = np.nan_to_num(np.asarray(target_position, dtype=float))
    if len(target) != n:
        raise ValueError("длина сигнала не совпадает с числом тиков")

    # тик, на котором исполняется сигнал каждого тика
    exec_idx = np.searchsorted(t, t + int(latency_ms * 1_000_000), side="left")
    exec_idx = np.maximum(exec_idx, np.arange(1, n + 1))
    # если несколько сигналов попали на один тик исполнения, берётся последний
    # (промежуточные развороты внутри latency не учитываются)
    last_of_group = np.r_[exec_idx[1:] != exec_idx[:-1], True]
    take = last_of_group & (exec_idx < n)
    position = np.full(n, np.nan)
    position[exec_idx[take]] = target[take]
    position = pd.Series(position).ffill().fillna(0.0).to_numpy()

    mark = ticks["mid" if execution == "quote" else "price"].to_numpy()
    market_ret = np.zeros(n)
    market_ret[1:] = mark[1:] / mark[:-1] - 1

    prev_position = np.r_[0.0, position[:-1]]
    gross = prev_position * market_ret
    turnover = np.abs(position - prev_position)

    fees = turnover * fee_bps / 10_000
    spread_cost = turnover * ticks["half_spread"].to_numpy() / mark if execution == "quote" else np.zeros(n)
    slippage = turnover * slippage_bps / 10_000

    net = gross - fees - spread_cost - slippage
    equity = np.cumprod(1 + net)

    index = ticks.index
    held_ns = np.r_[np.diff(t), 0]  # сколько времени держится позиция после тика
    metrics = compute_metrics(pd.Series(equity, index=index), position, turnover,
                              fees, spread_cost, slippage, held_ns)
    return Result(equity=pd.Series(equity, index=index), returns=pd.Series(net, index=index),
                  position=pd.Series(position, index=index), metrics=metrics)


def compute_metrics(equity, position, turnover, fees, spread_cost, slippage, held_ns):
    # Sharpe по минутной доходности: тики неравномерны, годовой пересчёт по тикам некорректен
    eq_1m = equity.resample("1min").last().ffill()
    r = eq_1m.pct_change().dropna()
    std = r.std()
    sharpe = r.mean() / std * np.sqrt(MINUTES_PER_YEAR) if len(r) > 1 and std > 0 else 0.0

    eq = equity.to_numpy()
    max_dd = (eq / np.maximum.accumulate(eq) - 1).min()
    total_ns = held_ns.sum()
    time_in_market = (held_ns * (position != 0)).sum() / total_ns if total_ns > 0 else 0.0

    return {
        "total_return_%": round(float(eq[-1] - 1) * 100, 2),
        "sharpe_annual": round(float(sharpe), 2),
        "max_drawdown_%": round(float(max_dd) * 100, 2),
        "trades": int((turnover > 0).sum()),
        "fees_paid_%": round(float(fees.sum()) * 100, 2),
        "spread_paid_%": round(float(spread_cost.sum()) * 100, 2),
        "slippage_paid_%": round(float(slippage.sum()) * 100, 2),
        "time_in_market_%": round(float(time_in_market) * 100, 1),
    }


def train_test_split(ticks, train_frac=0.7):
    """Деление по времени (а не по числу тиков: активность в течение суток неравномерна)."""
    cut = ticks.index[0] + (ticks.index[-1] - ticks.index[0]) * train_frac
    return ticks[ticks.index < cut], ticks[ticks.index >= cut]
