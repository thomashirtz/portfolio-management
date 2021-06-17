from binance import Client

from portfolio_management.api import get_kline_dataframe

from portfolio_management.database.initialization import Initialization
from portfolio_management.database.insert import Insert

from portfolio_management.database.retrieve import get_dataframe
from portfolio_management.database.retrieve import get_dataset


def test_database_retrieve():
    folder_path = 'D:\\Thomas\\Python\\gym-portfolio-2\\databases'
    database_name = 'test'

    symbol_list = ["ETHBTC", "BNBBTC"]
    interval = Client.KLINE_INTERVAL_30MINUTE
    start = "2017-11-12"
    end = "2017-11-14"

    initialization = Initialization(folder_path=folder_path, database_name=database_name)
    initialization.run(symbol_list=symbol_list, reset_tables=True)
    insert = Insert(folder_path=folder_path, database_name=database_name)

    for symbol in symbol_list:
        data = get_kline_dataframe(symbol=symbol, interval=interval, start=start, end=end)
        insert.run(symbol=symbol, data=data)

    dataframe = get_dataframe(
        folder_path=folder_path,
        database_name=database_name,
        symbol=symbol_list[0]
    )

    dataset = get_dataset(
        folder_path=folder_path,
        database_name=database_name,
    )


if __name__ == '__main__':
    test_database_retrieve()