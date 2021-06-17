import pandas as pd

from portfolio_management.database.bases import Time
from portfolio_management.database.bases import Data

from portfolio_management.database.retrieve import get_symbol_id
from portfolio_management.database.retrieve import get_time_mapping
from portfolio_management.database.retrieve import get_property_mapping

from portfolio_management.database.utilities import session_scope
from portfolio_management.database.utilities import get_sessionmaker
from portfolio_management.database.utilities import silent_bulk_insert


class Insert:
    def __init__(
            self,
            folder_path: str,
            database_name: str,
            echo: bool = False
    ):
        self.Session = get_sessionmaker(folder_path, database_name, echo)

    def run(self, symbol: str, data: pd.DataFrame) -> None:

        with session_scope(self.Session) as session:
            self._insert_datetime(session, data)
            self._insert_data(session, symbol, data)

    @staticmethod
    def _insert_datetime(session, data: pd.DataFrame) -> None:
        datetime_set = set(data['open_time'])
        datetime_set.update(set(data['close_time']))
        instances = [Time(value=datetime) for datetime in datetime_set]
        silent_bulk_insert(session, instances)

    @staticmethod
    def _insert_data(session, symbol: str, data: pd.DataFrame) -> None:
        time_mapping = get_time_mapping(session=session)
        property_mapping = get_property_mapping(session=session)
        symbol_id = get_symbol_id(session=session, symbol=symbol)

        data['open_time_id'] = data["open_time"].map(time_mapping)
        data['close_time_id'] = data["close_time"].map(time_mapping)

        instances = []
        for _, row in data.iterrows():
            for property_name, property_id in property_mapping.items():
                instance = Data(
                    value=row[property_name],
                    symbol_id=symbol_id,
                    property_id=property_id,
                    open_time_id=row['open_time_id'],
                    close_time_id=row['close_time_id'],
                )
                instances.append(instance)
        silent_bulk_insert(session=session, instances=instances)

