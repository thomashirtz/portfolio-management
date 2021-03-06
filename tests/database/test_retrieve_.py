from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import get_dataset
from portfolio_management.data.retrieve import get_dataframe


def test_database_retrieve(
        print_dataframe: bool = True,
        print_dataset: bool = True
):

    folder_path = None
    database_name = 'test'

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

    dataframe = get_dataframe(
        folder_path=folder_path,
        database_name=database_name,
        symbol=symbol_list[0],
        interval=interval,
    )
    if print_dataframe:
        print(dataframe)

    dataset = get_dataset(
        folder_path=folder_path,
        database_name=database_name,
        interval=interval
    )
    if print_dataset:
        print(dataset)


if __name__ == '__main__':
    test_database_retrieve()
