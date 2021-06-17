from typing import List
from typing import Type
from typing import Optional

import pandas as pd
import xarray as xr

from sqlalchemy.orm import Session

from portfolio_management.database.bases import Time
from portfolio_management.database.bases import Base
from portfolio_management.database.bases import Data
from portfolio_management.database.bases import Symbol
from portfolio_management.database.bases import Property

from portfolio_management.database.utilities import session_scope
from portfolio_management.database.utilities import get_sessionmaker
from portfolio_management.database.utilities import remove_keys_from_dictionary


def get_symbol_id(session: Session, symbol: str) -> int:
    return session.query(Symbol.id).filter(Symbol.name == symbol).first()[0]


def get_symbol_list(session: Session) -> list:
    symbol_tuples = session.query(Symbol.name).all()
    return [symbol_tuple[0] for symbol_tuple in symbol_tuples]


def _get_mapping(session: Session, base: Type[Base], key_attr: str, value_attr: str):
    instances = list(session.query(base))
    return {getattr(instance, key_attr): getattr(instance, value_attr) for instance in instances}


def get_property_mapping(session: Session, id_to_name: bool = False):
    key_attr = 'id' if id_to_name else 'name'
    value_attr = 'name' if id_to_name else 'id'
    return _get_mapping(session, Property, key_attr=key_attr, value_attr=value_attr)


def get_symbol_mapping(session: Session, id_to_name: bool = False):
    key_attr = 'id' if id_to_name else 'name'
    value_attr = 'name' if id_to_name else 'id'
    return _get_mapping(session, Symbol, key_attr=key_attr, value_attr=value_attr)


def get_time_mapping(session: Session, id_to_value: bool = False):
    key_attr = 'id' if id_to_value else 'value'
    value_attr = 'value' if id_to_value else 'id'
    return _get_mapping(session, Time, key_attr=key_attr, value_attr=value_attr)


def get_dataframe(
        folder_path: str,
        database_name: str,
        symbol: str,
        echo: bool = False,
) -> pd.DataFrame:

    with session_scope(
            get_sessionmaker(folder_path, database_name, echo),
            expire_on_commit=False
    ) as session:

        symbol_id = get_symbol_id(session=session, symbol=symbol)
        property_mapping = get_property_mapping(session=session, id_to_name=True)
        instances = session.query(Data).filter(Data.symbol_id == symbol_id).all()

        records = {}
        for instance in instances:
            if instance.close_time_id not in records.keys():
                property_name = property_mapping[instance.property_id]
                dictionary = remove_keys_from_dictionary(
                    instance.__dict__,
                    ['_sa_instance_state', 'value', 'property_id', 'id', 'symbol_id']
                )
                dictionary[property_name] = instance.value
                records[instance.close_time_id] = dictionary
            else:
                property_name = property_mapping[instance.property_id]
                records[instance.close_time_id][property_name] = instance.value
        dataframe = pd.DataFrame.from_dict(records, orient='index')

        time_mapping = get_time_mapping(session=session, id_to_value=True)
        dataframe['open_time'] = dataframe["open_time_id"].map(time_mapping)
        dataframe['close_time'] = dataframe["close_time_id"].map(time_mapping)
        dataframe.drop(['open_time_id', 'close_time_id'], axis=1, inplace=True)
        # todo need to setup index column
    return dataframe


def get_dataset(
        folder_path: str,
        database_name: str,
        symbol_list: Optional[List[str]] = None,
        echo: bool = False,
) -> xr.Dataset:
    with session_scope(
            get_sessionmaker(folder_path, database_name, echo),
            expire_on_commit=False
    ) as session:
        symbol_list = symbol_list or get_symbol_list(session=session)  # noqa

    ds_list = []
    for symbol in symbol_list:
        df = get_dataframe(
            folder_path=folder_path,
            database_name=database_name,
            symbol=symbol,
        )
        ds_list.append(xr.Dataset.from_dataframe(df))
    ds = xr.concat(ds_list, dim='symbol')
    ds['symbol'] = symbol_list

    # ds['trades'] = ds['trades'] + epsilon
    return ds
