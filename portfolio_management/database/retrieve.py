from typing import List
from typing import Optional

import pandas as pd
import xarray as xr
import numpy as np

from sqlalchemy.orm import Session

import portfolio_management.database.constants as c

from portfolio_management.database.bases import Data
from portfolio_management.database.bases import Symbol
from portfolio_management.database.bases import Interval

from portfolio_management.database.utilities import session_scope
from portfolio_management.database.utilities import get_sessionmaker


def get_symbol_id(session: Session, symbol: str) -> int:
    return session.query(Symbol.id).filter(Symbol.name == symbol).first()[0]


def get_interval_id(session: Session, interval: str) -> int:
    return session.query(Interval.id).filter(Interval.value == interval).first()[0]


def get_symbol_list(session: Session) -> list:
    symbol_tuples = session.query(Symbol.name).all()
    return [symbol_tuple[0] for symbol_tuple in symbol_tuples]


def get_dataframe(
        folder_path: str,
        database_name: str,
        symbol: str,
        interval: str,
        echo: bool = False,
) -> pd.DataFrame:

    with session_scope(
            get_sessionmaker(folder_path, database_name, echo),
            expire_on_commit=False
    ) as session:

        symbol_id = get_symbol_id(session=session, symbol=symbol)
        interval_id = get_interval_id(session=session, interval=interval)

        instances = session.query(Data).filter(
            Data.symbol_id == symbol_id,
            Data.interval_id == interval_id,
        ).all()

        records = []
        for instance in instances:
            records.append(instance.__dict__)
        dataframe = pd.DataFrame(records)

        dataframe.drop([
            'id',
            'symbol_id',
            'interval_id',
            '_sa_instance_state',
        ], axis=1, inplace=True)

        dataframe.sort_values(by=[c.OPEN_TIME], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

    return dataframe


def get_dataset(
        folder_path: str,
        database_name: str,
        interval: str,
        symbol_list: Optional[List[str]] = None,
        echo: bool = False,
        float_32: bool = True
) -> xr.Dataset:
    with session_scope(
            get_sessionmaker(folder_path, database_name, echo),
            expire_on_commit=False
    ) as session:
        symbol_list = symbol_list or get_symbol_list(session=session)  # noqa

    open_time_array_list = []
    close_time_array_list = []
    properties_array_list = []
    for symbol in symbol_list:
        df = get_dataframe(
            folder_path=folder_path,
            database_name=database_name,
            symbol=symbol,
            interval=interval,
        )

        open_time_array_list.append(df[c.OPEN_TIME])
        close_time_array_list.append(df[c.CLOSE_TIME])
        properties_array_list.append(df[c.PROPERTY_LIST])

    dtype = 'float32' if float_32 else 'float64'
    properties_array = np.stack(properties_array_list).astype(dtype)

    open_time_array = np.stack(open_time_array_list)
    close_time_array = np.stack(close_time_array_list)
    indexes = np.arange(open_time_array.shape[1])

    dataset = xr.Dataset(
        {
            c.DATA: ([c.SYMBOL, c.INDEX, c.PROPERTY], properties_array),
            c.CLOSE_TIME: ([c.SYMBOL, c.INDEX], open_time_array),
            c.OPEN_TIME: ([c.SYMBOL, c.INDEX], close_time_array),
        },
        coords={
            c.SYMBOL: symbol_list,
            c.PROPERTY: c.PROPERTY_LIST,
            c.INDEX: indexes,
        },
        attrs={
            c.INTERVAL: interval
        }
    )
    return dataset
