import os
import sys
scripts_directory_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_directory_path + '/../')

from portfolio_management.database.manager import Manager


if __name__ == '__main__':
    folder_path = 'D:\\Thomas\\GitHub\\portfolio-management\\databases'
    database_name = 'database_2'

    symbol_list = [
        'BTCTUSD',
        'ETHTUSD',
        'BNBTUSD',
        'XRPTUSD',
    ]

    interval = '5m'
    start = "2020-01-01"
    end = "2020-12-31"

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