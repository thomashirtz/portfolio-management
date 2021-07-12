from typing import Optional
from typing import Union

import torch
import torch.nn as nn


def apply_along_axis(function, x, axis: int = 0):
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)


class GlobalAveragePooling1D(nn.Module):
    def __init__(self, dim: Optional[Union[int, tuple, list]]):
        super(GlobalAveragePooling1D, self).__init__()
        self.dim = dim

    def forward(self, x) -> torch.Tensor:
        return torch.mean(x, dim=self.dim)


class Evaluator(nn.Module):
    def __init__(
            self,
            num_observations,
            num_properties,
            num_currencies,
            num_intermediate_outputs,
            num_outputs
    ):
        super(Evaluator, self).__init__()
        self.num_properties = num_properties
        self.num_observations = num_observations
        self.num_outputs = num_outputs
        self.num_currencies = num_currencies
        self.num_intermediate_outputs = num_intermediate_outputs


class BasicEvaluator(Evaluator):
    def __init__(
            self,
            num_observations,
            num_properties,
            num_outputs,
            num_currencies,
            num_intermediate_outputs
    ):
        super(BasicEvaluator, self).__init__(
            num_observations=num_observations,
            num_properties=num_properties,
            num_outputs=num_outputs,
            num_currencies=num_currencies,
            num_intermediate_outputs=num_intermediate_outputs,
        )

        self.independent_evaluator = nn.Linear(num_properties * num_observations, num_intermediate_outputs)
        self.linear = nn.Linear(num_currencies * num_intermediate_outputs + num_currencies + 2, num_outputs)

    def forward(
            self,
            market_observation: torch.Tensor,
            portfolio_observation: torch.Tensor,
            proportion_observation: torch.Tensor
    ) -> torch.Tensor:
        observation = torch.flatten(market_observation, start_dim=2)
        output = apply_along_axis(self.independent_evaluator, observation)
        concatenate = torch.cat([output, portfolio_observation, proportion_observation], dim=0)
        return self.linear(concatenate)



