import pandas as pd
from binance import Client


def get_kline_dataframe(symbol: str, interval: str, start: str, end: str):
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'ignore'
    ]

    client = Client()
    klines = client.get_historical_klines(symbol, interval, start_str=start, end_str=end)
    # todo check how much data we can download at the time

    dataframe = pd.DataFrame(klines, columns=columns)
    dataframe.index = dataframe['close_time']

    dataframe['open_time'] = pd.to_datetime(dataframe['open_time'] * 1_000_000)
    dataframe['close_time'] = pd.to_datetime((dataframe['close_time'] + 1) * 1_000_000)
    dataframe.drop(columns=['ignore'], inplace=True)
    return dataframe


interval_to_seconds = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600,
    '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800, '12h': 43200,
    '1d': 86400, '3d': 259200, '1w': 604800, '1M': 0,  # todo check one month value
}
