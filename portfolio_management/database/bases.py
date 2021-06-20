from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

import portfolio_management.database.constants as c


Base = declarative_base()


class Symbol(Base):
    __tablename__ = "symbol"

    id = Column('id', Integer, primary_key=True)
    name = Column('name', String, unique=True)

    data = relationship("Data", backref="symbol", cascade_backrefs=False)

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, name={self.name})"


class Interval(Base):
    __tablename__ = "interval"

    id = Column('id', Integer, primary_key=True)
    value = Column('value', String, unique=True)

    data = relationship("Data", backref="interval", cascade_backrefs=False)

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, value={self.value})"


class Data(Base):
    __tablename__ = "data"

    id = Column('id', Integer, primary_key=True)

    open_time = Column('open_time', DateTime, nullable=False)
    close_time = Column('close_time', DateTime, nullable=False)

    open = Column('open', Float)
    high = Column('high', Float)
    low = Column('low', Float)
    close = Column('close', Float)
    volume = Column('volume', Float)

    number_of_trades = Column('number_of_trades', Float)
    quote_asset_volume = Column('quote_asset_volume', Float)
    taker_buy_base_asset_volume = Column('taker_buy_base_asset_volume', Float)
    taker_buy_quote_asset_volume = Column('taker_buy_quote_asset_volume', Float)

    symbol_id = Column('symbol_id', Integer, ForeignKey('symbol.id'))
    interval_id = Column('interval_id', Integer, ForeignKey('interval.id'))

    def get_properties(self):
        return {p: getattr(self, p) for p in c.PROPERTY_LIST}

    def __repr__(self):
        return f"{type(self).__name__}(" \
               f"id={self.id}, " \
               f"open_time={self.open_time}, " \
               f"close_time={self.close_time}, " \
               f"open={self.open}, " \
               f"high={self.high}, " \
               f"low={self.low}, " \
               f"close={self.close}, " \
               f"volume={self.volume}, " \
               f"quote_asset_volume={self.quote_asset_volume}, " \
               f"number_of_trades={self.number_of_trades}, " \
               f"taker_buy_base_asset_volume={self.taker_buy_base_asset_volume}, " \
               f"taker_buy_quote_asset_volume={self.taker_buy_quote_asset_volume}, " \
               f"symbol_id={self.symbol_id}, " \
               f"interval_id={self.interval_id}, " \
               f")"
