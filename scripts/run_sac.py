import os
import sys
scripts_directory_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_directory_path + '/../')


import gym
import portfolio_management
from portfolio_management import soft_actor_critic


if __name__ == '__main__':  # can't test sac yet, there is 'inf' value in datasets

    # first run 'prepare_pickled_dataset' !
    # todo find why it is slow

    train = False
    dataset_name = 'USDT_train'

    env_kwargs = {
        'dataset_name': dataset_name,
        'num_steps': 100,
        'fees': 0.005,
        'seed': 1,
        'step_size': 1,
        'chronologically': False,
        'observation_size': 50,
        'stake_range': [100, 100],
    }

    # SAC kwargs
    memory_size = 1_000_000
    batch_size = 128
    updates_per_step = 3
    learning_rate = 3e-4
    alpha = 0.05
    gamma = 1
    tau = 0.005
    num_steps = 10_000
    start_step = 0

    if train:
        soft_actor_critic.train(
            'Portfolio-v0',
            env_kwargs=env_kwargs,

            num_steps=num_steps,
            learning_rate=learning_rate,
            updates_per_step=updates_per_step,
            batch_size=batch_size,
            start_step=start_step,

            alpha=alpha,
            gamma=gamma,
            tau=tau,

            memory_size=memory_size,
        )

    else:
        dataset_name = 'USDT_test'
        env_kwarg_test = {
            'dataset_name': dataset_name,
            'num_steps': None,
        }
        env_kwargs.update(**env_kwarg_test)
        run_name = '20210720_192852_Portfolio-v0'
        soft_actor_critic.render(
            'Portfolio-v0',
            run_name=run_name,
            env_kwargs=env_kwargs,
            num_episodes=1
        )

