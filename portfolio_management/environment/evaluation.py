import gym
import numpy as np
from typing import Optional

from portfolio_management.utilities import get_str_time


def evaluate_hold(
        holding_values: list,
        num_episodes: int = 500,
        env_name: str = 'Portfolio-v0',
        env_kwargs: Optional[dict] = None,
):

    env_kwargs = env_kwargs or {}
    env = gym.make(env_name, **env_kwargs)

    score_history = []

    for episode in range(num_episodes):
        score = 0
        done = False
        episode_step = 0
        _ = env.reset()

        while not done:
            new_observation, reward, done, info = env.step(holding_values)
            score += reward
            episode_step += 1

        score_history.append(score)
        print(
            f'\rEpisode n°{episode}  Steps: {episode_step} '
            f'\tScore: {score:.3f} \tMean: {np.mean(score_history):.3f}'
            f'\tTime: {get_str_time(env.market.current_time)}'
        )
