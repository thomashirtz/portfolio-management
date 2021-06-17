from math import ceil
from typing import Optional

import numpy as np
import xarray as xr

from portfolio_management.data import get_dataset, sample_dataset


class Market:
    def __init__(
            self,
            folder_path: str,
            currencies: list,
            config: dict,
            max_steps: int,
            timestep_per_step: int = 1,
            apply_log: bool = True,
            chronologically: bool = True,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ):
        assert timestep_per_step <= config[max(config.keys())], \
            'Impossible to do more timestep than the largest timestep\'s step'

        self.folder_path = folder_path
        self.currencies = currencies
        self.timestep_to_number = config
        self.max_steps = max_steps
        self.timestep_per_step = timestep_per_step
        self.apply_log = apply_log
        self.chronologically = chronologically
        self.end_date = end_date
        self.start_date = start_date

        self.datasets = {}
        self.main_sample = None

        for time_step, number in sorted(self.timestep_to_number.items()):
            suffix = f'USD_{time_step}.csv'
            dataset = get_dataset(
                currencies,
                suffix=suffix,
                folder_path=folder_path,
                start_date=start_date,
                end_date=end_date
            )
            self.datasets[time_step] = dataset

        self.main_timestep = max(self.timestep_to_number.keys())
        self.constraint = ceil(max([ts*n for ts, n in self.timestep_to_number.items()]) / self.main_timestep)
        self.current_time = self.datasets[self.main_timestep]['time'][self.constraint]

    def reset(self):
        if not self.chronologically:
            self.current_time = np.random.choice(self.datasets[self.main_timestep]['time'][self.constraint: - self.constraint])
        if self.current_time > self.datasets[self.main_timestep]['time'][-self.constraint]:
            self.current_time = self.datasets[self.main_timestep]['time'][self.constraint]
            print('restart from zero')

    def step(self):
        raw_observation = []
        for time_step, number in sorted(self.timestep_to_number.items()):
            sample = sample_dataset(self.datasets[time_step], number, self.current_time)
            raw_observation.append(self._flatten(sample))
            if time_step == self.main_timestep:
                self.main_sample = sample

        self.current_time += self.main_timestep * self.timestep_per_step * 60
        open = np.array(self.main_sample.isel(time=-self.timestep_per_step)['open'])
        close = np.array(self.main_sample.isel(time=-1)['close'])
        observation = np.concatenate(raw_observation, axis=None)

        if self.apply_log:
            observation = np.log(observation)

        return observation, open, close

    @staticmethod
    def _flatten(dataset: xr.Dataset) -> np.array:
        return np.array(dataset.to_array()).flatten()

    def __repr__(self):
        return f'<{self.__class__.__name__}('\
               f'folder_path={self.folder_path!r},' \
               f'currencies={self.currencies!r},' \
               f'config={self.timestep_to_number!r},' \
               f'max_steps={self.max_steps!r},' \
               f'timestep_per_step={self.timestep_per_step!r}, ' \
               f'apply_log={self.apply_log!r}, ' \
               f'chronologically={self.chronologically!r}, ' \
               f'start_date={self.start_date!r}, ' \
               f'end_date={self.end_date!r})>'
