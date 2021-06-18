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

    dataframe = pd.DataFrame(klines, columns=columns)
    dataframe.index = dataframe['close_time']

    dataframe['open_time'] = pd.to_datetime(dataframe['open_time'] * 1_000_000)
    dataframe['close_time'] = pd.to_datetime((dataframe['close_time'] + 1) * 1_000_000)
    dataframe.drop(columns=['ignore'], inplace=True)
    return dataframe
