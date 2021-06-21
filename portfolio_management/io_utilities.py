import yaml
import json
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


def read_json_file(file_path) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json_file(file_path, data: dict) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f)
