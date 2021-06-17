import datetime
import numpy as np
from pathlib import Path


def get_unix_time(date: str):
    time = datetime.datetime.strptime(date, '%Y%m%d')
    return int((time - datetime.datetime(1970, 1, 1)).total_seconds())


def get_str_time(unix_time):
    return datetime.datetime.fromtimestamp(int(unix_time)).strftime('%Y%m%d')


def loguniform(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def rate_calculator(amount, principal, time):
    return np.log(amount/principal) / time


def create_folders(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
