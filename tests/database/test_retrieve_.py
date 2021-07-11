from binance import Client

from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import get_dataset
from portfolio_management.data.retrieve import get_dataframe


def test_database_retrieve():

    folder_path = 'D:\\Thomas\\GitHub\\portfolio-management\\databases'
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

    dataframe = get_dataframe(
        folder_path=folder_path,
        database_name=database_name,
        symbol=symbol_list[0],
        interval=interval,
    )

    dataset = get_dataset(
        folder_path=folder_path,
        database_name=database_name,
        interval=interval
    )


if __name__ == '__main__':
    test_database_retrieve()
