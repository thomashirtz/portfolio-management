import yaml
import pickle
from pathlib import Path


def pickle_dump(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_folders(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_yaml(path_yaml_file: Path, dictionary: dict):
    with open(path_yaml_file, 'w') as f:
        yaml.dump(dictionary, f, sort_keys=False)
