from binance import Client

from portfolio_management.data.manager import Manager


def test_database_manager():
    folder_path = None
    database_name = 'test'

    symbol_list = ["ETHBTC"]
    interval = Client.KLINE_INTERVAL_30MINUTE
    start = "2017-11-12"
    end = "2017-11-14"
    manager = Manager(
        folder_path=folder_path,
        database_name=database_name,
        echo=False,
        reset_tables=True
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=start,
        end=end,
    )


if __name__ == '__main__':
    test_database_manager()
