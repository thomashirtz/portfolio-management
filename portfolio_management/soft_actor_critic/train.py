from typing import Optional
from typing import Sequence

import gym
import torch
import datetime
import numpy as np
from pathlib import Path
from itertools import count

from torch.utils.tensorboard import SummaryWriter

from portfolio_management.soft_actor_critic.agent import Agent
from portfolio_management.soft_actor_critic.utilities import filter_info
from portfolio_management.soft_actor_critic.utilities import filter_info_step
from portfolio_management.soft_actor_critic.utilities import get_run_name
from portfolio_management.soft_actor_critic.utilities import save_to_writer
from portfolio_management.soft_actor_critic.utilities import get_timedelta_formatted


def train(
        env_name: str,
        env_kwargs: Optional[dict] = None,
        module: Optional[torch.nn.Module] = None,
        batch_size: int = 256,
        memory_size: int = 10e6,
        learning_rate: float = 3e-4,
        alpha: float = 0.05,
        gamma: float = 0.99,
        tau: float = 0.005,
        num_steps: int = 1_000_000,
        hidden_units: Optional[Sequence[int]] = None,
        load_models: bool = False,
        saving_frequency: int = 20,
        run_name: Optional[str] = None,
        start_step: int = 1_000,
        seed: int = 0,
        updates_per_step: int = 1,
        directory: str = '../runs/',
        **kwargs
):

    if kwargs:
        print(f'Unrecognized kwargs : {kwargs}')

    env_kwargs = env_kwargs or {}
    env = gym.make(env_name, **env_kwargs)

    observation_shape = env.observation_space.shape[0]  # todo learn how to handle 2D observations
    num_actions = env.action_space.shape[0]

    run_name = run_name if run_name is not None else get_run_name(env_name)
    run_directory = Path(directory) / run_name
    writer = SummaryWriter(run_directory)

    agent = Agent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        alpha=alpha,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        hidden_units=hidden_units,
        batch_size=batch_size,
        memory_size=memory_size,
        checkpoint_directory=run_directory,
        load_models=load_models
    )

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_training_time = datetime.datetime.now()
    print(f'Start training time: {start_training_time.strftime("%Y-%m-%d %H:%M:%S")}')

    updates = 0
    global_step = 0
    last_save_episode = -1
    score_history = []

    for episode in count():
        info = {}
        score = 0
        done = False
        episode_step = 0

        observation = env.reset()

        while not done:
            if start_step > global_step:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(observation)

            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)

            score += reward
            global_step += 1
            episode_step += 1
            observation = new_observation

            if agent.memory.memory_counter >= batch_size:
                for update in range(updates_per_step):
                    tensorboard_logs = agent.learn()
                    save_to_writer(writer, tensorboard_logs, updates)
                    updates += 1

            if global_step % 2 == 0:
                save_to_writer(writer, filter_info_step(info), global_step)

        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        time_delta = get_timedelta_formatted(datetime.datetime.now() - start_training_time)
        print(f'\r{time_delta}   [{global_step}/{num_steps}]   Episode n°{episode}   '
              f'Steps: {episode_step} \tScore: {score:.3f} \tAverage100: {average_score:.3f} \t'
              f'(Last save: Episode n°{last_save_episode})', end="", flush=True)

        tensorboard_logs = {
            'train/episode_step': episode_step,
            'train/score': score,
            'train/average_score': average_score,
            **filter_info(info)
        }
        save_to_writer(writer, tensorboard_logs, global_step)

        if episode % saving_frequency == 0:
            last_save_episode = episode
            agent.save_models()

        if global_step > num_steps:
            break
