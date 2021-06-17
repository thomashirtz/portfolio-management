from portfolio_management.database.initialization import Initialization


def test_database_initialization():
    folder_path = 'D:\\Thomas\\Python\\gym-portfolio-2\\databases'
    symbol_list = ['ETH', 'BTC']

    initialization = Initialization(folder_path=folder_path, database_name='test')
    initialization.run(symbol_list=symbol_list)


if __name__ == '__main__':
    test_database_initialization()