from pathlib import Path
from typing import Union
from typing import List
from typing import Optional

import portfolio_management.paths as p
from portfolio_management.io_utilities import pickle_load
from portfolio_management.data.retrieve import get_dataset as _get_dataset
from portfolio_management.data.preprocessing import get_pca_preprocessing_function


def get_dataset(
        name: str,
        currencies: Optional[List[str]] = None,
        databases_folder_path: Optional[Union[str, Path]] = None,
        datasets_folder_path: Optional[Union[str, Path]] = None,
):
    try:
        folder_path = p.get_datasets_folder_path(datasets_folder_path)  # todo need to change how it retrieve data
        file_path = folder_path.joinpath(name).with_suffix('.pkl')
        return pickle_load(file_path)
    except Exception as e:
        print(e) # todo check if processing_property is in dataset

    path = p.get_databases_folder_path(databases_folder_path)
    return _get_dataset(
        folder_path=str(path),
        database_name=name,
        symbol_list=currencies,
        preprocessing=get_pca_preprocessing_function()  # todo to remove, do not want to mix db et env
    )
