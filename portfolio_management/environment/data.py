from pathlib import Path
from typing import Union
from typing import Optional
import pandas as pd
import xarray as xr

from portfolio_management.utilities import get_unix_time


# todo add function to download the data


def get_dataframe(path: Union[str, Path]):
    columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'trades']
    dataframe = pd.read_csv(path, names=columns, index_col='time')
    dataframe['time'] = dataframe.index
    return dataframe


def get_dataset(
        currencies,
        suffix: str,
        folder_path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        epsilon: float = 10 ** -10,
):
    ds_list = []
    for currency in currencies:
        path = Path(folder_path) / (currency + suffix)
        df = get_dataframe(path)
        ds_list.append(xr.Dataset.from_dataframe(df))
    ds = xr.concat(ds_list, dim='currency', join='inner')
    ds['currency'] = currencies
    if start_date:
        ds = ds.where(get_unix_time(start_date) < ds.time, drop=True)
    if end_date:
        ds = ds.where(ds.time < get_unix_time(end_date), drop=True)

    ds['volume'] = ds['volume'] + epsilon
    ds['trades'] = ds['trades'] + epsilon
    return ds


def sample_dataset(dataset: xr.Dataset, sample_size: int, time: int):
    return dataset.sel(time=slice(None, time)).tail(time=sample_size)
