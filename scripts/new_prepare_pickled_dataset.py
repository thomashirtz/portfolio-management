from portfolio_management.data.manager import Manager
from portfolio_management.data.preprocessing import PCAPreprocessing
from portfolio_management.data.dataset import DatasetManager


def create_database(database_name, symbol_list, interval, start, end):
    manager = Manager(
        database_name=database_name,
        reset_tables=True,
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=start,
        end=end,
    )


if __name__ == '__main__':

    download_train_data = False
    download_test_data = False

    symbol_list = [
        'USDCUSDT',
        'BTCUSDT',
        'ETHUSDT',
        'BNBUSDT',
        'LTCUSDT',
        'XRPUSDT'
    ]
    interval = '1h'

    train_database_name = 'USDT_train'
    start = "2020-01-01"
    end = "2020-12-31"
    if download_train_data:
        create_database(train_database_name, symbol_list, interval, start, end)

    test_database_name = 'USDT_test'
    start = "2021-01-01"
    end = "2021-05-01"
    if download_test_data:
        create_database(test_database_name, symbol_list, interval, start, end)

    preprocessing_class = PCAPreprocessing()

    dataset_manager = DatasetManager(
        train_database_name=train_database_name,
        test_database_name=test_database_name,
    )
    dataset_manager.run(preprocessing_class)
