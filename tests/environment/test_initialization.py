import gym
import portfolio_management


def test_initialization():
    env_kwargs = {
        'database_name': 'database_1'
    }
    env = gym.make('Portfolio-v0', **env_kwargs)
    env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(f'State: {state} Reward: {reward} Done: {done}')


if __name__ == '__main__':
    test_initialization()
