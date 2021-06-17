from binance import Client

from portfolio_management.api import get_kline_dataframe

from portfolio_management.database.initialization import Initialization
from portfolio_management.database.insert import Insert


def test_database_insert():
    folder_path = 'D:\\Thomas\\GitHub\\portfolio-management\\databases'
    database_name = 'test'

    symbol_list = ["ETHBTC"]
    interval = Client.KLINE_INTERVAL_30MINUTE
    start = "2017-11-12"
    end = "2017-11-14"

    initialization = Initialization(folder_path=folder_path, database_name=database_name)
    initialization.run(symbol_list=symbol_list, reset_tables=True)
    insert = Insert(folder_path=folder_path, database_name=database_name)

    for symbol in symbol_list:
        data = get_kline_dataframe(symbol=symbol, interval=interval, start=start, end=end)
        insert.run(symbol=symbol, data=data)


if __name__ == '__main__':
    test_database_insert()