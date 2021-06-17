from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Symbol(Base):
    __tablename__ = "symbol"

    id = Column('id', Integer, primary_key=True)
    name = Column('name', String)
    # measurements = relationship("Data", backref="symbol", cascade_backrefs=False)

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, name={self.name})"


class Time(Base):
    __tablename__ = "time"

    id = Column('id', Integer, primary_key=True)
    value = Column('value', DateTime)
    # data = relationship("Data", backref="datetime", cascade_backrefs=False)
    #  todo check why it errors
    #   https://docs.sqlalchemy.org/en/13/orm/join_conditions.html#handling-multiple-join-paths

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, value={self.value})"


class Property(Base):
    __tablename__ = "property"

    id = Column('id', Integer, primary_key=True)
    name = Column('name', String)
    # data = relationship("Data", backref="property", cascade_backrefs=False)

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, name={self.name})"


class Data(Base):
    __tablename__ = "data"

    id = Column('id', Integer, primary_key=True)
    value = Column('value', Float)

    symbol_id = Column('symbol_id', Integer, ForeignKey('symbol.id'))
    property_id = Column('property_id', Integer, ForeignKey('property.id'))

    open_time_id = Column(Integer, ForeignKey('time.id'))
    close_time_id = Column(Integer, ForeignKey('time.id'))

    open_time = relationship("Time", foreign_keys=[open_time_id])
    close_time = relationship("Time", foreign_keys=[close_time_id])

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, name={self.name}, symbol_id={self.symbol_id}, property_id={self.property_id}, open_time_id={self.open_time_id}, close_time_id={self.close_time_id})"  # noqa
