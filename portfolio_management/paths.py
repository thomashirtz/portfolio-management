from pathlib import Path
from typing import Union
from typing import Optional


project_path = Path(__file__).parent.parent
runs_folder_path = project_path.joinpath('runs')
datasets_folder_path = project_path.joinpath('datasets')
databases_folder_path = project_path.joinpath('databases')


def _get_path(default_path: Path, path: Optional[Union[str, Path]] = None) -> Path:
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        return default_path


def get_runs_folder_path(path: Optional[Union[str, Path]] = None) -> Path:
    return _get_path(path, runs_folder_path)


def get_datasets_folder_path(path: Optional[Union[str, Path]] = None) -> Path:
    return _get_path(path, datasets_folder_path)


def get_databases_folder_path(path: Optional[Union[str, Path]] = None) -> Path:
    return _get_path(path, databases_folder_path)
