import os
import sys
scripts_directory_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_directory_path + '/../')

from portfolio_management.database.manager import Manager
from portfolio_management.database.retrieve import pickle_database

if __name__ == '__main__':
    database_name = 'database_0'

    symbol_list = [
        'BTCTUSD',
        'ETHTUSD',
        'BNBTUSD',
        'XRPTUSD',
    ]

    interval = '5m'
    start = "2020-01-01"
    end = "2020-01-02"

    manager = Manager(
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
    pickle_database(database_name=database_name)
