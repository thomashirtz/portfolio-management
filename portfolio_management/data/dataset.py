from typing import List
from typing import Optional

import numpy as np
import xarray as xr

import portfolio_management.paths as p
import portfolio_management.data.constants as c
from portfolio_management.io_utilities import pickle_dump

from portfolio_management.data.bases import Interval
from portfolio_management.data.retrieve import get_dataframe
from portfolio_management.data.retrieve import get_symbol_list

from portfolio_management.data.utilities import session_scope
from portfolio_management.data.utilities import get_sessionmaker

from portfolio_management.data.preprocessing import Preprocessing


class DatasetManager:  # todo edit the data structure as well as the folder structure because it is not coherent yet
    def __init__(
            self,
            train_database_name: str,
            test_database_name: Optional[str] = None,
            dataset_folder_path: Optional[str] = None,
            database_folder_path: Optional[str] = None,
            interval: Optional[str] = None,
            symbol_list: Optional[List[str]] = None,
            echo: bool = False,
            float_32: bool = True,
    ):

        self.databases_folder_path = p.get_databases_folder_path(database_folder_path)
        self.datasets_folder_path = p.get_datasets_folder_path(dataset_folder_path)
        self.train_database_name = train_database_name
        self.test_database_name = test_database_name
        self.echo = echo
        self.float_32 = float_32

        with session_scope(
                get_sessionmaker(str(self.databases_folder_path), self.train_database_name, echo),
                expire_on_commit=False
        ) as session:
            self.symbol_list = symbol_list or get_symbol_list(session=session)  # noqa
            self.interval = interval or session.query(Interval.value).first()[0]

    def run(self, preprocessing: Preprocessing, pickle: bool = True, float_32: bool = True):
        train_dataset = self.get_dataset(self.train_database_name, preprocessing, pickle, is_test=False, float_32=float_32)
        test_dataset = self.get_dataset(self.test_database_name, preprocessing, pickle, is_test=True, float_32=float_32)
        return train_dataset, test_dataset

    def get_dataset(
            self,
            database_name: str,
            preprocessing: Preprocessing,
            pickle: bool,
            is_test: bool = False,
            float_32: bool = True,
    ):
        dataframe_dict = {}
        open_time_array_dict = {}
        close_time_array_dict = {}
        properties_array_dict = {}

        for symbol in self.symbol_list:
            dataframe = get_dataframe(
                folder_path=str(self.databases_folder_path),
                database_name=database_name,
                symbol=symbol,
                interval=self.interval,
            )

            dataframe_dict[symbol] = dataframe
            open_time_array_dict[symbol] = dataframe[c.OPEN_TIME]
            close_time_array_dict[symbol] = dataframe[c.CLOSE_TIME]
            properties_array_dict[symbol] = dataframe[c.PROPERTY_LIST]

        preprocessed_array_dict = {}  # todo it is not really an array
        if preprocessing is not None:
            preprocessed_array_dict = preprocessing(dataframe_dict, is_test=is_test)

        dtype = 'float32' if float_32 else 'float64'
        properties_array = np.stack([properties_array_dict[symbol] for symbol in self.symbol_list]).astype(dtype)

        open_time_array = np.stack([open_time_array_dict[symbol] for symbol in self.symbol_list])
        close_time_array = np.stack([close_time_array_dict[symbol] for symbol in self.symbol_list])
        indexes = np.arange(open_time_array.shape[1])

        data_preprocessing = {}
        coords_preprocessing = {}
        if preprocessing is not None:
            coords = list(list(preprocessed_array_dict.values())[0].columns)
            coords_preprocessing = {c.PREPROCESSING_PROPERTY: coords}  # edit for having the thing in k
            preprocessed_data = np.stack([preprocessed_array_dict[symbol] for symbol in self.symbol_list])
            data_preprocessing = {
                c.DATA_PREPROCESSED: ([c.SYMBOL, c.INDEX, c.PREPROCESSING_PROPERTY], preprocessed_data)
            }

        dataset = xr.Dataset(
            {
                c.DATA: ([c.SYMBOL, c.INDEX, c.PROPERTY], properties_array),
                c.CLOSE_TIME: ([c.SYMBOL, c.INDEX], open_time_array),  # todo put only one time in the dataset
                c.OPEN_TIME: ([c.SYMBOL, c.INDEX], close_time_array),
                **data_preprocessing
            },
            coords={
                c.SYMBOL: self.symbol_list,
                c.PROPERTY: c.PROPERTY_LIST,
                c.INDEX: indexes,
                **coords_preprocessing
            },
            attrs={
                c.INTERVAL: self.interval
            }
        )

        dataset = dataset.isel({c.INDEX: slice(0, -2)})

        if pickle:
            path_dataset = self.datasets_folder_path.joinpath(database_name).with_suffix('.pkl')
            pickle_dump(path_dataset, dataset)

        return dataset

