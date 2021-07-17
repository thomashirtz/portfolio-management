import numpy as np
import xarray as xr

from portfolio_management.data import constants as c


EPSILON = 10e-10  # avoid 0 in array before the applying log


class Market:  # todo add dimension
    def __init__(
            self,
            dataset: xr.Dataset,
            num_steps: int,
            observation_size: int,
            step_size: int = 1,
            apply_log: bool = False,
            chronologically: bool = True,
    ):
        self.dataset = dataset

        self.num_steps = num_steps
        self.step_size = step_size
        self.observation_size = observation_size

        self.apply_log = apply_log
        self.chronologically = chronologically

        self.margin_down = num_steps * step_size - 1 + self.observation_size
        self.margin_up = num_steps * step_size
        self.current_index = self.margin_down
        self.max_index = len(self.dataset[c.INDEX])

    def reset(self):
        if not self.chronologically:
            self.current_index = np.random.randint(self.margin_down, self.max_index - self.margin_up)
        # print(f'{self.margin_up=},{self.max_index=},{self.current_index=}')
        if self.current_index > self.max_index - self.margin_up:
            self.current_index = self.margin_down
            print('restart from zero')

    def step(self):
        index_slice = slice(self.current_index - (self.observation_size - 1), self.current_index)  # todo - 1 is a hack need to recheck
        observation = self.dataset[c.DATA_PREPROCESSED].sel({c.INDEX: index_slice})

        open_ = np.array(self.dataset[c.DATA].sel({c.PROPERTY: 'open'}).isel({c.INDEX: -self.step_size}))
        close = np.array(self.dataset[c.DATA].sel({c.PROPERTY: 'close'}).isel({c.INDEX: -1}))

        if self.apply_log:
            observation = np.log(observation + EPSILON)

        self.current_index += 1
        return self.dataset_to_numpy(observation), open_, close

    @property
    def current_time(self):
        value = self.dataset[c.OPEN_TIME].isel({c.SYMBOL: 0})[-1].values  # todo remove isel when only one datatime in dataset
        return int(np.datetime_as_string(value, unit='D').replace('-', ''))

    @staticmethod
    def dataset_to_numpy(dataset: xr.Dataset) -> np.array:
        array = np.array(dataset)
        # Change array from (Symbol, Index, Property) to (Symbol, Property, Index) for better pytorch handling
        array_reordered = np.moveaxis(array, [1, 2], [2, 1])
        return array_reordered

