from binance import Client
from portfolio_management.binance_api import get_kline_dataframe


def test_get_kline_dataframe():
    symbol = "ETHBTC"
    interval = Client.KLINE_INTERVAL_30MINUTE
    start = "2017-11-12"
    end = "2017-11-14"
    df = get_kline_dataframe(symbol=symbol, interval=interval, start=start, end=end)


if __name__ == '__main__':
    test_get_kline_dataframe()
