from typing import List
from pathlib import Path
from sqlalchemy import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from portfolio_management.io_utilities import write_yaml
from portfolio_management.io_utilities import create_folders

from portfolio_management.database.bases import Base
from portfolio_management.database.bases import Data
from portfolio_management.database.bases import Symbol
from portfolio_management.database.bases import Interval

from portfolio_management.binance_api import get_kline_dataframe

from portfolio_management.database.retrieve import get_symbol_id
from portfolio_management.database.retrieve import get_interval_id

from portfolio_management.database.utilities import session_scope
from portfolio_management.database.utilities import get_engine_url
from portfolio_management.database.utilities import try_insert
from portfolio_management.database.utilities import get_path_database
from portfolio_management.database.utilities import silent_bulk_insert


class Manager:
    def __init__(
            self,
            folder_path: str,
            database_name: str,
            echo: bool = False,
            reset_tables: bool = False,
    ):
        self.database_name = database_name
        self.folder_path = folder_path

        create_folders(get_path_database(folder_path, database_name).parent)

        self.engine_url = get_engine_url(folder_path, database_name)
        self.engine = create_engine(self.engine_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)

        if reset_tables:
            Base.metadata.drop_all(bind=self.engine)

        if self.database_is_empty(self.engine) or reset_tables:
            Base.metadata.create_all(bind=self.engine)

    def insert(
            self,
            symbol_list: List[str],
            interval: str,
            start: str,
            end: str,
    ):
        for symbol in symbol_list:
            dataframe = get_kline_dataframe(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end
            )
            self._insert_dataframe(
                symbol=symbol,
                interval=interval,
                dataframe=dataframe,
            )

        config = {
            "folder_path": self.folder_path,
            "database_name": self.database_name,
            "symbol_list": symbol_list,
            "interval": interval,
            "start": start,
            "end": end,
        }
        path_yaml_file = Path(self.folder_path).joinpath(self.database_name).with_suffix('.yaml')
        write_yaml(path_yaml_file=path_yaml_file, dictionary=config)
        print('Config saved')

    def _insert_dataframe(self, symbol, interval, dataframe):
        with session_scope(self.Session) as session:
            try_insert(session, Symbol, key='name', value=symbol)
            try_insert(session, Interval, key='value', value=interval)
            symbol_id = get_symbol_id(session, symbol=symbol)
            interval_id = get_interval_id(session, interval=interval)

        instances = []
        for _, row in dataframe.iterrows():
            instance = Data(
                symbol_id=symbol_id,
                interval_id=interval_id,
                **dict(row)
            )
            instances.append(instance)

        silent_bulk_insert(session=session, instances=instances)

    @staticmethod
    def database_is_empty(engine):
        table_names = inspect(engine).get_table_names()
        is_empty = table_names == []
        return is_empty
