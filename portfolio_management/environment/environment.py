from typing import Tuple
from typing import Optional

from gym import Env
from gym import spaces
from gym.utils import seeding

import numpy as np
from scipy.special import softmax

from portfolio_management.market import Market
from portfolio_management.portfolio import Portfolio
from portfolio_management.utilities import rate_calculator
from portfolio_management.utilities import get_str_time


DEFAULT_PRINCIPAL_RANGE = [10, 1000]
DEFAULT_TIMESTEP_TO_STEP = {60: 12, 15: 4}
DEFAULT_CURRENCIES = ['ETH', 'XBT', 'USDT', 'XRP', 'XDG']


class PortfolioEnv(Env):  # noqa
    def __init__(
            self,
            currencies: Optional[list] = None,
            num_steps: int = 100,
            fees: float = 0.002,
            seed: Optional[int] = None,
            chronologically: bool = True,
            folder_path: str = '..//Kraken_OHLCVT',
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            config: Optional[dict] = None,
            timestep_per_step: Optional[int] = 1,
            principal_range: Optional[list] = None,
    ):

        self.seed(seed)
        self.current_step = None
        self.num_steps = num_steps
        self.timestep_per_step = timestep_per_step
        self.config = config or DEFAULT_TIMESTEP_TO_STEP
        self.currencies = currencies or DEFAULT_CURRENCIES
        self.principal_range = principal_range or DEFAULT_PRINCIPAL_RANGE

        self.portfolio = Portfolio(
            self.currencies,
            fees=fees,
            principal_range=self.principal_range,
        )

        self.market = Market(
            folder_path=folder_path,
            currencies=self.currencies,
            config=self.config,
            max_steps=num_steps,
            timestep_per_step=timestep_per_step,
            apply_log=True,
            chronologically=chronologically,
            start_date=start_date,
            end_date=end_date,
        )

        prod = sum(self.config.values())
        state = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.currencies),)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.currencies) * len(state) * prod + len(self.currencies) + 2,)
        )

    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0
        portfolio_state = self.portfolio.reset()
        self.market.reset()
        market_state, _, _ = self.market.step()  # todo check should we count -1
        return np.concatenate((market_state, portfolio_state))

    def step(self, action: list) -> Tuple[list, float, bool, dict]:
        if sum(action) == 1:
            proportions = np.array(action)
        else:
            proportions = softmax(action)

        self.current_step += 1
        done = True if self.current_step >= self.num_steps else False
        market_state, open_, close = self.market.step()
        reward, _, portfolio_state = self.portfolio.step(proportions, open_, close)

        state = np.concatenate((market_state, portfolio_state))
        time = self.current_step * self.market.main_timestep * self.market.timestep_per_step / (60 * 24)
        rate = rate_calculator(self.portfolio.amount, self.portfolio.principal, time)

        info = {}
        for i, currency in enumerate(self.currencies):
            info[f'step:portfolio/{currency}'] = float(proportions[i])

        info['information/date'] = get_str_time(self.market.current_time)

        info['information/amount'] = self.portfolio.amount
        info['information/principal'] = self.portfolio.principal

        info['portfolio/monthly_interest'] = np.exp(1) ** (rate * 30.5) - 1
        info['portfolio/yearly_interest'] = np.exp(1) ** (rate * 365) - 1
        info['portfolio/current_interest'] = self.portfolio.amount / self.portfolio.principal - 1

        return state, reward, done, info
