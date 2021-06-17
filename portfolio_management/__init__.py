from gym.envs.registration import register

register(
    id='Portfolio-v0',
    entry_point='portfolio_management.environment:PortfolioEnv',
)
