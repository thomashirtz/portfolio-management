from typing import Union
from typing import List
import numpy as np

from portfolio_management.environment.utilities import loguniform


class Portfolio:
    def __init__(
            self,
            currencies: list,
            fees: float,
            principal_range: List[float]
    ):
        self.amount = None
        self.principal = None

        self.proportions = None

        self.fees = fees
        self.currencies = currencies
        self.principal_range = principal_range

    def reset(self):
        self.proportions = None
        self.amount = self.principal = loguniform(*self.principal_range)
        return self.state

    def step(
            self,
            new_proportions: Union[list, np.array],
            open_: Union[list, np.array],
            close: Union[list, np.array],
    ):
        if self.proportions is None:
            fees = 0
        else:
            fees = np.sum(np.abs(np.array(new_proportions) - self.proportions) * self.amount * self.fees)

        self.proportions = np.array(new_proportions)
        values = self.proportions * self.amount
        growth = np.array(close) / np.array(open_)
        new_values = values * growth
        new_amount = np.sum(new_values)

        reward = (new_amount - self.amount - fees) / self.amount * 100  # todo maybe use principal if not stable
        self.amount = new_amount - fees

        return reward, self.amount, self.state

    @property
    def state(self) -> np.array:
        proportions = self.proportions if self.proportions is not None else np.zeros(shape=len(self.currencies))
        state = np.concatenate((
            [np.log(self.amount)],
            [np.log(self.principal)],
            proportions,
        ))
        return state

    def __repr__(self):
        return f'<{self.__class__.__name__}('\
               f'currencies={self.currencies!r},' \
               f'fees={self.fees}, ' \
               f'principal_range={self.principal_range})>'

    def __str__(self):
        return f'<{self.__class__.__name__} '\
               f'amount={self.amount}, ' \
               f'principal={self.principal}, ' \
               f'proportions={self.proportions})>'
