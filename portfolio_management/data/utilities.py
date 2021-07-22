from typing import Type
from typing import List
from pathlib import Path

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from portfolio_management.data.bases import Base


def get_path_database(folder_path: str, database_name: str) -> Path:
    return Path(folder_path).joinpath(database_name).with_suffix('.db')


def get_engine_url(folder_path: str, database_name: str) -> str:
    path_database = get_path_database(folder_path, database_name)
    return r'sqlite:///' + str(path_database)


def get_sessionmaker(
        folder_path: str,
        database_name: str,
        echo: bool = False,
        timeout: int = 60
) -> sessionmaker:
    engine_url = get_engine_url(folder_path, database_name)
    engine = create_engine(engine_url, echo=echo, connect_args={'timeout': timeout})
    return sessionmaker(bind=engine)


@contextmanager
def session_scope(session_maker: sessionmaker, **kwargs) -> None:
    """Provide a transactional scope around a series of operations."""
    session = session_maker(**kwargs)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def silent_insert(session: Session, instance: Type[Base]) -> None:
    try:
        session.add(instance)
        session.commit()
    except SQLAlchemyError as e:
        print('silent_insert_exception', e)


def silent_bulk_insert(session: Session, instances: List[Type[Base]]) -> None:
    try:
        session.bulk_save_objects(instances)
        session.commit()
    except SQLAlchemyError as e:
        print('silent_bulk_insert_exception', e)


def try_insert(session: Session, base: Type[Base], key: str, value: str) -> None:
    count = session.query(getattr(base, 'id')).filter(getattr(base, key) == value).count()
    if not count:
        instance = base(**{key: value})
        silent_insert(session, instance)


def find_instance(
        session: Session,
        base: Base,
        column_to_value: dict
) -> Type[Base]:
    criteria = (getattr(base, column).like(value) for column, value in column_to_value.items())
    return session.query(base).filter(*criteria).first()


def inner_join(a, b) -> list:
    return list(set(a) & set(b))


def remove_keys_from_dictionary(dictionary: dict, keys: list) -> dict:
    return {k: v for k, v in dictionary.items() if k not in keys}

def prepare_dataset():
    pass
