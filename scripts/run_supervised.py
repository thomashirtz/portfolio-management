import os
import sys
scripts_directory_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_directory_path + '/../')

# I will in the future segment that into different modules and clean it up, but I would like to first know where I'm going

from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import pickle_database

from portfolio_management.io_utilities import pickle_load
from portfolio_management.paths import datasets_folder_path

download = False
train_database_name = 'database_train'
test_database_name = 'database_test'

if download:
    symbol_list = [
        'BTCTUSD',  # https://www.tradingview.com/symbols/BTCTUSD/
        'ETHTUSD',
        'BNBTUSD',
        'XRPTUSD',
    ]

    interval = '5m'

    train_start = "2020-01-01"
    train_end = "2020-31-12"

    test_start = "2021-01-01"
    test_end = "2021-06-01"

    manager = Manager(
        database_name=train_database_name,
        echo=False,
        reset_tables=True
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=train_start,
        end=train_end,
    )
    print(f'{train_database_name} populated')
    pickle_database(database_name=train_database_name)

    manager = Manager(
        database_name=test_database_name,
        echo=False,
        reset_tables=True
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=test_start,
        end=test_end,
    )
    print(f'{test_database_name} populated')
    pickle_database(database_name=test_database_name)


path_train_dataset = datasets_folder_path.joinpath(train_database_name).with_suffix('.pkl')
train_dataset = pickle_load(path_train_dataset)

path_train_dataset = datasets_folder_path.joinpath(train_database_name).with_suffix('.pkl')
train_dataset = pickle_load(train_dataset)

print()
