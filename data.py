"""Загрузка тиковых данных Binance (aggTrades).

Формат тиков во всём проекте: DataFrame с DatetimeIndex "time" (UTC, ns,
отсортирован, дубликаты времени допустимы) и колонками
    price  - цена сделки
    qty    - объём в базовой валюте
    side   - +1 агрессор покупатель (сделка по ask), -1 агрессор продавец (по bid)
"""
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VISION_URL = "https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
API_URL = "https://api.binance.com/api/v3/aggTrades"
CACHE_DIR = Path(__file__).parent / "data_cache"

AGG_COLUMNS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "is_best_match"]


def make_ticks(time_raw, price, qty, is_buyer_maker):
    """Собирает DataFrame тиков из сырых массивов Binance."""
    time_raw = np.asarray(time_raw, dtype=np.int64)
    # с 2025 года спотовые дампы data.binance.vision в микросекундах, API отдаёт миллисекунды
    unit = "us" if len(time_raw) and time_raw.max() > 10**14 else "ms"
    index = pd.DatetimeIndex(pd.to_datetime(time_raw, unit=unit), name="time").as_unit("ns")
    ticks = pd.DataFrame({
        "price": np.asarray(price, dtype=np.float64),
        "qty": np.asarray(qty, dtype=np.float64),
        "side": np.where(np.asarray(is_buyer_maker, dtype=bool), -1, 1).astype(np.int8),
    }, index=index)
    return ticks.sort_index(kind="stable")


def _to_bool(col):
    if col.dtype == bool:
        return col.to_numpy()
    return col.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def read_vision_zip(content):
    """Разбирает дневной zip aggTrades с data.binance.vision (с заголовком или без)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            has_header = not f.readline().decode().strip()[:1].isdigit()
        with zf.open(name) as f:
            raw = pd.read_csv(
                f, header=None, names=AGG_COLUMNS, skiprows=1 if has_header else 0,
                usecols=["price", "qty", "time", "is_buyer_maker"],
            )
    return make_ticks(raw["time"], raw["price"], raw["qty"], _to_bool(raw["is_buyer_maker"]))


def _download_day(symbol, day):
    import requests

    url = VISION_URL.format(symbol=symbol, day=day)
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        raise FileNotFoundError(f"{url} не найден (файл за день появляется на data.binance.vision на следующие сутки)")
    resp.raise_for_status()
    content = resp.content

    checksum = requests.get(url + ".CHECKSUM", timeout=30)
    if checksum.ok:
        expected = checksum.text.split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if expected != actual:
            raise ValueError(f"контрольная сумма не совпала для {url}")
    return read_vision_zip(content)


def download_ticks(symbol="BTCUSDT", days=1, end_date=None):
    """Тики за `days` полных суток UTC, заканчивая end_date (по умолчанию вчера).

    Каждый день скачивается одним zip-файлом и кэшируется в data_cache/.
    """
    end = date.fromisoformat(end_date) if end_date else datetime.now(timezone.utc).date() - timedelta(days=1)
    CACHE_DIR.mkdir(exist_ok=True)

    parts = []
    for i in range(days - 1, -1, -1):
        day = (end - timedelta(days=i)).isoformat()
        cache_file = CACHE_DIR / f"{symbol}_aggTrades_{day}.pkl"
        if cache_file.exists():
            part = pd.read_pickle(cache_file)
        else:
            print(f"скачиваю {symbol} {day} ...", flush=True)
            part = _download_day(symbol, day)
            part.to_pickle(cache_file)
        parts.append(part)
    return pd.concat(parts).sort_index(kind="stable")


def download_ticks_api(symbol="BTCUSDT", hours=1.0, sleep=0.05):
    """Последние `hours` часов тиков через REST API (без кэша, удобно для свежих данных).

    API отдаёт по 1000 сделок за запрос, поэтому для суток и больше быстрее download_ticks.
    """
    import requests

    def get(params):
        resp = requests.get(API_URL, params={"symbol": symbol, **params}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(hours * 3_600_000)

    # первый id сделки в окне; дальше пагинация по fromId
    first = get({"startTime": start_ms, "endTime": min(start_ms + 3_599_999, end_ms), "limit": 1})
    if not first:
        raise RuntimeError(f"нет сделок {symbol} в первый час периода")
    from_id = first[0]["a"]

    rows = []
    requests_done = 0
    while True:
        batch = get({"fromId": from_id, "limit": 1000})
        requests_done += 1
        if not batch:
            break
        rows.extend(b for b in batch if b["T"] <= end_ms)
        if batch[-1]["T"] > end_ms or len(batch) < 1000:
            break
        from_id = batch[-1]["a"] + 1
        if requests_done % 100 == 0:
            print(f"  {len(rows):,} сделок ...", flush=True)
        time.sleep(sleep)

    return make_ticks(
        [r["T"] for r in rows], [float(r["p"]) for r in rows],
        [float(r["q"]) for r in rows], [r["m"] for r in rows],
    )


def synthetic_ticks(n=200_000, seed=42, mean_dt=0.25, half_spread_bps=0.5):
    """Синтетические тики: пуассоновский поток сделок, mid с возвратом к медленному среднему,
    сделки по bid/ask с персистентным направлением агрессора и небольшим импактом."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.cumsum(rng.exponential(mean_dt, n)), unit="s")

    flips = rng.random(n) < 0.3
    noise = rng.normal(0, 2e-5, n)
    side = np.empty(n, dtype=np.int8)
    log_mid = np.empty(n)
    side[0], log_mid[0] = 1, np.log(60_000)
    anchor = log_mid[0]
    for i in range(1, n):
        side[i] = -side[i - 1] if flips[i] else side[i - 1]
        log_mid[i] = log_mid[i - 1] + 1e-5 * side[i] + noise[i] - 0.002 * (log_mid[i - 1] - anchor)
        anchor += 0.001 * (log_mid[i] - anchor)

    price = np.exp(log_mid) * (1 + side * half_spread_bps / 10_000)
    ticks = pd.DataFrame({
        "price": price,
        "qty": rng.lognormal(-4, 1, n),
        "side": side,
    }, index=pd.DatetimeIndex(t, name="time").as_unit("ns"))
    return ticks
