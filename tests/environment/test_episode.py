import gym
import portfolio_management
# import datetime


def test_episode():
    env_kwargs = {
        'database_name': 'test_feature'
    }
    env = gym.make('Portfolio-v0', **env_kwargs) # todo not pass by make to be able to debug
    env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        print(f'State: {state} Reward: {reward} Done: {done}')


if __name__ == '__main__':
    test_episode()