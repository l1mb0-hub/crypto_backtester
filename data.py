from pathlib import Path
import time

import numpy as np
import pandas as pd

BINANCE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = Path(__file__).parent / "data_cache"


def download_klines(symbol="BTCUSDT", interval="1m", days=7):
    import requests

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{days}d.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col="time", parse_dates=True)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    rows = []
    while start_ms < end_ms:
        resp = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000},
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated()]
    df.to_csv(cache_file)
    return df


def synthetic_klines(n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    log_price = np.zeros(n)
    log_price[0] = np.log(60_000)
    for i in range(1, n):
        noise = rng.normal(0, 0.001)
        pull_back = -0.02 * (log_price[i - 1] - log_price[max(0, i - 60):i].mean())
        log_price[i] = log_price[i - 1] + noise + pull_back
    close = np.exp(log_price)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": rng.uniform(1, 10, n),
    }, index=idx)
