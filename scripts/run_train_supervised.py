from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import pickle_database
from portfolio_management.io_utilities import pickle_load
from portfolio_management.paths import datasets_folder_path

database_name = 'database_0'

if not datasets_folder_path.joinpath(database_name).with_suffix('.pkl').exists():
    symbol_list = [
        'BTCTUSD',
        'ETHTUSD',
        'XRPTUSD',
    ]

    interval = '5m'
    start = "2020-01-01"
    end = "2020-01-02"

    manager = Manager(
        database_name=database_name,
        echo=False,
        reset_tables=True,
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=start,
        end=end,
    )
    pickle_database(database_name=database_name)

path = datasets_folder_path.joinpath(database_name).with_suffix('.pkl')
dataset = pickle_load(path)
