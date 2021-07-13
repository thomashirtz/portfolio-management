from typing import Tuple
from typing import Optional

from gym import Env
from gym import spaces
from gym.utils import seeding

import numpy as np
from scipy.special import softmax

from portfolio_management import paths as p
from portfolio_management.data import constants as c

from portfolio_management.environment.data import get_dataset
from portfolio_management.environment.market import Market
from portfolio_management.environment.portfolio import Portfolio
# from portfolio_management.environment.utilities import rate_calculator


DEFAULT_STAKE_RANGE = [10, 1000]


class PortfolioEnv(Env):  # noqa
    def __init__(
            self,
            database_name: str,
            currencies: Optional[list] = None,
            num_steps: int = 100,
            fees: float = 0.002,
            seed: Optional[int] = None,
            chronologically: bool = False,
            step_size: Optional[int] = 1,
            observation_size: int = 25,
            stake_range: Optional[list] = None,
            databases_folder_path: Optional[str] = None,
            datasets_folder_path: Optional[str] = None,
    ):

        datasets_folder_path = datasets_folder_path or p.datasets_folder_path
        databases_folder_path = databases_folder_path or p.databases_folder_path
        self.dataset = get_dataset(  # todo edit so that we accept only pickle to avoid ambiguity and complexity
            name=database_name,
            currencies=currencies,
            datasets_folder_path=datasets_folder_path,
            databases_folder_path=databases_folder_path,
        )

        self.currencies = list(self.dataset[c.SYMBOL].values)

        self.seed(seed)
        self.current_step = None
        self.num_steps = num_steps
        self.step_size = step_size
        self.stake_range = stake_range or DEFAULT_STAKE_RANGE

        self.portfolio = Portfolio(
            currencies=self.currencies,
            fees=fees,
            principal_range=self.stake_range,
        ) # todo edit portfolio state

        self.market = Market(
            dataset=self.dataset,
            apply_log=False,  # todo remove epsilon et apply log, remove preprocessing from this part of the project
            step_size=step_size,
            num_steps=num_steps,
            chronologically=chronologically,
            observation_size=observation_size,
        )

        num_features = len(self.dataset[c.PREPROCESSING_PROPERTY].values)

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.currencies), )
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(  # Market state
                low=-np.inf,
                high=np.inf,
                shape=(len(self.currencies), num_features, observation_size)
            ),
            spaces.Box(  # Current portfolio proportions
                low=0,
                high=1,
                shape=(len(self.currencies), )
            ),
            spaces.Box(  # Principal and Amount
                low=np.log(min(self.stake_range)),
                high=np.log(max(self.stake_range)),
                shape=(2,)),
        ))

    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0
        portfolio_state = self.portfolio.reset()
        self.market.reset()
        market_dataset, _, _ = self.market.step()
        market_state = np.array(market_dataset.values).flatten()  # todo issue with -inf present in dataset
        return np.concatenate((market_state, portfolio_state))

    def step(self, action: list) -> Tuple[list, float, bool, dict]:
        if sum(action) == 1:
            proportions = np.array(action)
        else:
            proportions = softmax(action)

        self.current_step += 1
        done = True if self.current_step >= self.num_steps else False
        market_dataset, open_, close = self.market.step()
        reward, _, portfolio_state = self.portfolio.step(proportions, open_, close)

        market_state = np.array(market_dataset.values).flatten()  # todo check issue with market state -inf
        state = np.concatenate((market_state, portfolio_state))

        # If the key in 'info' contains '/' it would be added to tensorboard every end of episode (done=True)
        # If the key in 'info' starts by 'step:' the value will be recorded every step instead of every episode
        info = {}
        for i, currency in enumerate(self.currencies):
            info[f'step:portfolio/{currency}'] = float(proportions[i])

        info['information/date'] = self.market.current_time
        info['information/amount'] = self.portfolio.amount
        info['information/principal'] = self.portfolio.principal

        # todo update the following code
        # time = self.current_step * self.market.main_timestep * self.market.step_size / (60 * 24)
        # rate = rate_calculator(self.portfolio.amount, self.portfolio.principal, time)
        #
        # info['portfolio/monthly_interest'] = np.exp(1) ** (rate * 30.5) - 1
        # info['portfolio/yearly_interest'] = np.exp(1) ** (rate * 365) - 1
        # info['portfolio/current_interest'] = self.portfolio.amount / self.portfolio.principal - 1

        return state, reward, done, info
