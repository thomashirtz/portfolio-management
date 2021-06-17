from binance import Client
from portfolio_management.database.core import setup_database


def test_core():
    folder_path = 'D:\\Thomas\\Python\\gym-portfolio-2\\databases'
    database_name = 'test'

    symbol_list = ["ETHBTC", "BNBBTC"]
    interval = Client.KLINE_INTERVAL_30MINUTE
    start = "2017-11-12"
    end = "2017-11-14"

    setup_database(
        folder_path=folder_path,
        database_name=database_name,
        symbol_list=symbol_list,
        interval=interval,
        start=start,
        end=end,
    )


if __name__ == '__main__':
    test_core()