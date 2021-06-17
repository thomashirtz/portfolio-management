import yaml
from typing import List
from pathlib import Path

from portfolio_management.database.insert import Insert
from portfolio_management.database.initialization import Initialization
from portfolio_management.api import get_kline_dataframe


def setup_database(
        folder_path: str,
        database_name: str,
        symbol_list: List[str],
        interval: str,
        start: str,
        end: str,
):

    initialization = Initialization(folder_path=folder_path, database_name=database_name)
    initialization.run(symbol_list=symbol_list, reset_tables=True)
    insert = Insert(folder_path=folder_path, database_name=database_name)

    for symbol in symbol_list:
        data = get_kline_dataframe(symbol=symbol, interval=interval, start=start, end=end)
        insert.run(symbol=symbol, data=data)

    config = {
        "folder_path": folder_path,
        "database_name": database_name,
        "symbol_list": symbol_list,
        "interval": interval,
        "start": start,
        "end": end,
    }
    path_yaml_file = Path(folder_path).joinpath(database_name).with_suffix('.yaml')
    with open(path_yaml_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        print('config saved')
    print()