from typing import Tuple
from typing import Optional
from typing import Any

from gym import Env
from gym import spaces
from gym.utils import seeding

import numpy as np
from scipy.special import softmax

from portfolio_management import paths as p
from portfolio_management.data import constants as c

from portfolio_management.io_utilities import pickle_load
from portfolio_management.environment.market import Market
from portfolio_management.environment.portfolio import Portfolio
# from portfolio_management.environment.utilities import rate_calculator


DEFAULT_STAKE_RANGE = [10, 1000]


class PortfolioEnv(Env):  # noqa
    def __init__(
            self,
            dataset_name: str,
            num_steps: int = 100,
            fees: float = 0.005,
            seed: Optional[int] = None,
            chronologically: bool = False,
            step_size: Optional[int] = 1,
            observation_size: int = 25,
            stake_range: Optional[list] = None,
            datasets_folder_path: Optional[str] = None,
    ):

        datasets_folder_path = p.get_datasets_folder_path(datasets_folder_path)
        file_path = datasets_folder_path.joinpath(dataset_name).with_suffix('.pkl')
        self.dataset = pickle_load(file_path)

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
        )  # todo edit portfolio state

        self.market = Market(
            dataset=self.dataset,
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
                low=0,
                high=1,
                shape=(len(self.portfolio.state),)),
        ))

    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> Any:
        self.current_step = 0
        portfolio_state = self.portfolio.reset()
        proportions_state = self.portfolio.proportions
        self.market.reset()
        market_state, _, _ = self.market.step()
        return market_state, proportions_state, portfolio_state

    def step(self, action: list) -> Tuple[Any, float, bool, dict]:
        if sum(action) == 1:
            proportions = np.array(action)
        else:
            proportions = softmax(action)

        self.current_step += 1

        market_state, open_, close = self.market.step()
        reward, _, portfolio_state = self.portfolio.step(proportions, open_, close)

        proportions_state = self.portfolio.proportions
        state = market_state, proportions_state, portfolio_state

        if self.num_steps is None:
            done = self.market.done
        elif self.current_step >= self.num_steps:
            done = True
        else:
            done = False

        # If the key in 'info' contains '/' it would be added to tensorboard every end of episode (done=True)
        # If the key in 'info' starts by 'step:' the value will be recorded every step instead of every episode
        info = {}
        for i, currency in enumerate(self.currencies):
            info[f'step:portfolio/{currency}'] = float(proportions[i])

        info['information/date'] = self.market.current_time # todo only show one
        info['information/amount'] = self.portfolio.amount
        info['information/principal'] = self.portfolio.principal
        info['information/ratio_amount_principal'] = self.portfolio.amount/self.portfolio.principal

        return state, reward, done, info

    @property
    def render_data(self) -> dict:
        # return dict of all the close +
        # normalize to one
        # multiply by principal
        # add amount to dicy
        dictionary = self.market.data_render
        dictionary['amount'] = self.portfolio.amount
        return dictionary
