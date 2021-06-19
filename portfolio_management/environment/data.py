from pathlib import Path
from portfolio_management.io_utilities import pickle_load
from portfolio_management.database.retrieve import get_dataset as _get_dataset


def get_dataset(
        name: str,
        databases_folder_path: Path,
        datasets_folder_path: Path
):
    try:
        path = datasets_folder_path.joinpath(name).with_suffix('.pkl')
        return pickle_load(path)
    except Exception as e:
        print(e)

    return _get_dataset(folder_path=str(databases_folder_path), database_name=name)
