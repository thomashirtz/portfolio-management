import portfolio_management.paths as p
from portfolio_management.io_utilities import pickle_dump
from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import get_dataset
from portfolio_management.data.preprocessing import get_pca_preprocessing_function


if __name__ == '__main__':

    database_name = 'USDT_train'

    symbol_list = [
        'USDCUSDT',
        'BTCUSDT',
        'ETHUSDT',
        'BNBUSDT',
        'LTCUSDT',
        'XRPUSDT'
    ]
    interval = '1h'
    start = "2020-01-01"
    end = "2020-12-31"

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

    preprocessing_function = get_pca_preprocessing_function()

    dataset = get_dataset( # todo find a way to prepare the test dataset the same way as the train set
        database_name=database_name,
        interval=interval,
        preprocessing=preprocessing_function
    )

    datasets_folder_path = p.datasets_folder_path
    path_dataset = datasets_folder_path.joinpath(database_name).with_suffix('.pkl')
    pickle_dump(path_dataset, dataset)