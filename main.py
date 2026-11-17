import argparse
import itertools

import matplotlib.pyplot as plt

from backtest import run_backtest, train_test_split
from data import download_klines, synthetic_klines
from strategies import STRATEGIES, mean_reversion

PARAM_GRID = {"window": [30, 60, 120, 240], "entry_z": [1.5, 2.0, 2.5, 3.0]}


def optimize(train, test, fee_bps):
    results = []
    for window, entry_z in itertools.product(PARAM_GRID["window"], PARAM_GRID["entry_z"]):
        res = run_backtest(train, mean_reversion(train, window, entry_z), fee_bps)
        results.append((res.metrics["sharpe_annual"], window, entry_z))
    results.sort(reverse=True)

    print(f"{'window':>7} {'z':>5} {'sharpe_train':>13} {'sharpe_test':>12}")
    for sharpe_train, window, entry_z in results[:3]:
        res_test = run_backtest(test, mean_reversion(test, window, entry_z), fee_bps)
        print(f"{window:>7} {entry_z:>5} {sharpe_train:>13} {res_test.metrics['sharpe_annual']:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        df = synthetic_klines()
        name = "SYNTHETIC"
    else:
        df = download_klines(args.symbol, "1m", args.days)
        name = args.symbol
    print(f"{name}: {len(df)} candles, {df.index[0]} - {df.index[-1]}")

    train, test = train_test_split(df)

    if args.optimize:
        optimize(train, test, args.fee_bps)
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test["close"] / test["close"].iloc[0], label="buy & hold", color="gray", alpha=0.6)

    print(f"test period, fee {args.fee_bps} bps")
    for strat_name, strat in STRATEGIES.items():
        res = run_backtest(test, strat(test), args.fee_bps)
        print(f"\n{strat_name}")
        for k, v in res.metrics.items():
            print(f"  {k}: {v}")
        ax.plot(res.equity, label=strat_name)

    if not args.no_plot:
        ax.set_title(f"{name} equity (test)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity.png", dpi=120)
        plt.show()


if __name__ == "__main__":
    main()
