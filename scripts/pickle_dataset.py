import xarray as xr
from portfolio_management.database.retrieve import pickle_database


if __name__ == '__main__':
    pickle_database(database_name='database_1')