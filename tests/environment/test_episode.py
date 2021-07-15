import gym
import portfolio_management
import portfolio_management.paths as p
from portfolio_management.io_utilities import pickle_dump


def test_episode(download_and_pickle: bool = True):
    database_name = 'test_episode'

    if download_and_pickle:
        from portfolio_management.data.manager import Manager
        from portfolio_management.data.retrieve import get_dataset
        from portfolio_management.data.preprocessing import get_pca_preprocessing_function

        symbol_list = ['BNBBTC', 'BNBETH']
        interval = '30m'
        start = "2020-01-01"
        end = "2020-02-01"

        manager = Manager(
            database_name=database_name,
            reset_tables=True,
        )
        manager.insert(
            symbol_list=symbol_list,
            interval=interval,
            start=start,
            end=end,
        )

        preprocessing_function = get_pca_preprocessing_function()

        dataset = get_dataset(
            database_name=database_name,
            interval=interval,
            preprocessing=preprocessing_function
        )

        datasets_folder_path = p.datasets_folder_path
        path_dataset = datasets_folder_path.joinpath(database_name).with_suffix('.pkl')
        pickle_dump(path_dataset, dataset)

    env_kwargs = {
        'dataset_name': database_name
    }
    env = gym.make('Portfolio-v0', **env_kwargs) # todo not pass by make to be able to debug
    env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        print(f'State: {state} Reward: {reward} Done: {done}')


if __name__ == '__main__':
    test_episode()