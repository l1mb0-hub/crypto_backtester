import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd

from backtest import add_quotes, run_backtest, train_test_split
from data import download_ticks, download_ticks_api, synthetic_ticks
from strategies import STRATEGIES, mean_reversion

PARAM_GRID = {"window": ["1min", "5min", "15min", "60min"], "entry_z": [1.5, 2.0, 2.5, 3.0]}


def optimize(train, test, bt_kwargs):
    results = []
    for window, entry_z in itertools.product(PARAM_GRID["window"], PARAM_GRID["entry_z"]):
        res = run_backtest(train, mean_reversion(train, window, entry_z), **bt_kwargs)
        results.append((res.metrics["sharpe_annual"], window, entry_z))
    results.sort(key=lambda r: r[0], reverse=True)

    print(f"{'window':>7} {'z':>5} {'sharpe_train':>13} {'sharpe_test':>12}")
    for sharpe_train, window, entry_z in results[:3]:
        res_test = run_backtest(test, mean_reversion(test, window, entry_z), **bt_kwargs)
        print(f"{window:>7} {entry_z:>5} {sharpe_train:>13} {res_test.metrics['sharpe_annual']:>12}")


def downsample(series, points=3000):
    """Для графика: миллионы тиков matplotlib рисует медленно."""
    span = series.index[-1] - series.index[0]
    freq = max(pd.Timedelta(span / points).ceil("1s"), pd.Timedelta("1s"))
    return series.resample(freq).last().dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=1, help="полных суток UTC с data.binance.vision")
    parser.add_argument("--end-date", default=None, help="последний день, YYYY-MM-DD (по умолчанию вчера)")
    parser.add_argument("--api-hours", type=float, default=None,
                        help="вместо дневных файлов взять последние N часов через REST API")
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="доп. проскальзывание на сделку")
    parser.add_argument("--latency-ms", type=float, default=50.0, help="задержка от сигнала до исполнения")
    parser.add_argument("--exec", dest="execution", choices=["quote", "trade"], default="quote",
                        help="quote: по mid с оплатой полуспреда; trade: по цене следующей сделки")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        ticks = synthetic_ticks()
        name = "SYNTHETIC"
    elif args.api_hours:
        ticks = download_ticks_api(args.symbol, args.api_hours)
        name = args.symbol
    else:
        ticks = download_ticks(args.symbol, args.days, args.end_date)
        name = args.symbol
    print(f"{name}: {len(ticks):,} ticks, {ticks.index[0]} - {ticks.index[-1]}")

    ticks = add_quotes(ticks)
    spread_bps = (2 * ticks["half_spread"] / ticks["mid"]).median() * 10_000
    print(f"median estimated spread: {spread_bps:.3f} bps")

    train, test = train_test_split(ticks)
    bt_kwargs = dict(fee_bps=args.fee_bps, latency_ms=args.latency_ms,
                     execution=args.execution, slippage_bps=args.slippage_bps)

    if args.optimize:
        optimize(train, test, bt_kwargs)
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    bh = test["mid"] / test["mid"].iloc[0]
    ax.plot(downsample(bh), label="buy & hold", color="gray", alpha=0.6)

    print(f"test period, fee {args.fee_bps} bps, latency {args.latency_ms} ms, exec {args.execution}")
    for strat_name, strat in STRATEGIES.items():
        res = run_backtest(test, strat(test), **bt_kwargs)
        print(f"\n{strat_name}")
        for k, v in res.metrics.items():
            print(f"  {k}: {v}")
        ax.plot(downsample(res.equity), label=strat_name)

    if not args.no_plot:
        ax.set_title(f"{name} equity (test)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity.png", dpi=120)
        plt.show()


if __name__ == "__main__":
    main()
