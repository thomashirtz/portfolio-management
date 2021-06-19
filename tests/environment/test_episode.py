import gym
import portfolio_management
# import datetime


def test_episode():
    env_kwargs = {
        'database_name': 'database_1'
    }
    env = gym.make('Portfolio-v0', **env_kwargs)
    env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        # print(f'State: {state} Reward: {reward} Done: {done}')


if __name__ == '__main__':
    test_episode()