import gym
import portfolio_management
import portfolio_management.paths as p
from portfolio_management.io_utilities import pickle_dump
from portfolio_management.soft_actor_critic import train


def test_episode(download_and_pickle: bool = False):
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
        'dataset_name': database_name,
        'num_steps': 100,
        'fees': 0.002,
        'seed': 1,
        'step_size': 1,
        'chronologically': False,
        'observation_size': 12,
        'stake_range': [100, 100],
    }

    memory_size = 1_000
    batch_size = 256
    updates_per_step = 1
    learning_rate = 3e-4
    alpha = 0.05
    gamma = 1
    tau = 0.005
    num_steps = 200

    train(
        'Portfolio-v0',
        env_kwargs=env_kwargs,

        num_steps=num_steps,
        learning_rate=learning_rate,
        updates_per_step=updates_per_step,
        batch_size=batch_size,

        alpha=alpha,
        gamma=gamma,
        tau=tau,

        memory_size=memory_size,
    )


if __name__ == '__main__':
    test_episode()
