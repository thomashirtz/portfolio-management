from typing import Optional
from typing import Type

from sqlalchemy import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from portfolio_management import utilities

from portfolio_management.database.bases import Base
from portfolio_management.database.bases import Symbol
from portfolio_management.database.bases import Property

from portfolio_management.database.utilities import session_scope
from portfolio_management.database.utilities import silent_insert
from portfolio_management.database.utilities import get_engine_url
from portfolio_management.database.utilities import get_path_database


class Initialization:
    def __init__(self, folder_path: str, database_name: str, echo=False):
        self.database_name = database_name
        self.folder_path = folder_path

        utilities.create_folders(get_path_database(folder_path, database_name).parent)

        self.engine_url = get_engine_url(folder_path, database_name)
        self.engine = create_engine(self.engine_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)

    def run(self, symbol_list: Optional[list], reset_tables=False) -> None:

        type_list = [
            'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume'
        ]

        if reset_tables:
            Base.metadata.drop_all(bind=self.engine)

        if self.database_is_empty(self.engine) or reset_tables:
            Base.metadata.create_all(bind=self.engine)
            with session_scope(self.Session) as session:
                self.table_initialization(session, Property, type_list)
                self.table_initialization(session, Symbol, symbol_list)

    @staticmethod
    def table_initialization(session, base: Type[Base], name_list: list) -> None:
        for name in name_list:
            instance = base(name=name)
            silent_insert(session, instance)

    @staticmethod
    def database_is_empty(engine):
        table_names = inspect(engine).get_table_names()
        is_empty = table_names == []
        return is_empty
