from pathlib import Path
from typing import Union
from typing import Optional

import portfolio_management.paths as p
from portfolio_management.io_utilities import pickle_load
from portfolio_management.database.retrieve import get_dataset as _get_dataset


def get_dataset(
        name: str,
        databases_folder_path: Optional[Union[str, Path]] = None,
        datasets_folder_path: Optional[Union[str, Path]] = None,
):
    try:
        folder_path = p.get_datasets_folder_path(datasets_folder_path)
        file_path = folder_path.joinpath(name).with_suffix('.pkl')
        return pickle_load(file_path)
    except Exception as e:
        print(e)

    path = p.get_databases_folder_path(databases_folder_path)
    return _get_dataset(folder_path=str(path), database_name=name)
