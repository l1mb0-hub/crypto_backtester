# crypto_backtester

Бэктестер для стратегий на минутных свечах Binance. Данные берутся через публичный API и кэшируются в csv.

Есть две стратегии: mean reversion по z-score и momentum на пересечении скользящих средних. Сигнал исполняется на следующей свече, комиссия списывается с каждого изменения позиции. Данные делятся на train/test по времени (70/30).

Метрики: доходность, годовой Sharpe, max drawdown, число сделок, сумма комиссий, время в позиции.

## Запуск

```
pip install -r requirements.txt
python main.py
python main.py --symbol ETHUSDT --days 14
python main.py --fee-bps 2
python main.py --optimize
python main.py --synthetic
```

`--optimize` перебирает параметры mean reversion на train и выводит sharpe лучших трех на test.

`--synthetic` запускает на сгенерированных данных, если нет доступа к Binance.

## Структура

- `data.py` загрузка данных
- `strategies.py` стратегии
- `backtest.py` бэктест и метрики
- `main.py` запуск

## TODO

- проскальзывание
- walk-forward
- order book imbalance по данным стакана
