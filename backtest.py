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


def run_backtest(df, target_position, fee_bps=4.0):
    position = target_position.shift(1).fillna(0)

    market_ret = df["close"].pct_change().fillna(0)
    gross = position * market_ret

    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * fee_bps / 10_000

    net = gross - costs
    equity = (1 + net).cumprod()

    return Result(equity=equity, returns=net, position=position,
                  metrics=compute_metrics(net, equity, turnover, costs))


def compute_metrics(net, equity, turnover, costs):
    std = net.std()
    sharpe = net.mean() / std * np.sqrt(MINUTES_PER_YEAR) if std > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return_%": round((equity.iloc[-1] - 1) * 100, 2),
        "sharpe_annual": round(sharpe, 2),
        "max_drawdown_%": round(max_dd * 100, 2),
        "trades": int((turnover > 0).sum()),
        "fees_paid_%": round(costs.sum() * 100, 2),
        "time_in_market_%": round((net != 0).mean() * 100, 1),
    }


def train_test_split(df, train_frac=0.7):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]
