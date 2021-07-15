from portfolio_management.data.manager import Manager
from portfolio_management.data.retrieve import get_dataset
from portfolio_management.data.preprocessing import get_pca_preprocessing_function


def test_database_retrieve(display_dataset: bool = True):
    folder_path = None
    database_name = 'test_preprocessing'

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

    preprocessing_function = get_pca_preprocessing_function()

    dataset = get_dataset(
        folder_path=folder_path,
        database_name=database_name,
        interval=interval,
        preprocessing=preprocessing_function
    )

    if display_dataset:
        print(dataset)


if __name__ == '__main__':
    test_database_retrieve()
