from portfolio_management.data.manager import Manager


def test_database_manager():
    folder_path = None
    database_name = 'test_manager'

    symbol_list = ["ETHBTC"]
    interval = '30m'
    start = "2020-01-01"
    end = "2020-02-01"

    manager = Manager(
        folder_path=folder_path,
        database_name=database_name,
        echo=False,
        reset_tables=True
    )
    manager.insert(
        symbol_list=symbol_list,
        interval=interval,
        start=start,
        end=end,
    )


if __name__ == '__main__':
    test_database_manager()
