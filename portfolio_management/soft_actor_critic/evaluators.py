from typing import Optional
from typing import Union
from typing import Sequence
from typing import List

import torch
import torch.nn as nn


# small extractor
# big evaluator

# other name: feature_extractor, evaluator, independent_evaluator, module


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


class Evaluator(nn.Module):  # todo change name
    def __init__(
            self,
            input_shapes: List[Union[Sequence[int], int]],
            num_intermediate_outputs: int,
            num_outputs: int,
    ):
        super(Evaluator, self).__init__()
        self.input_shapes = input_shapes
        self.num_outputs = num_outputs
        self.num_intermediate_outputs = num_intermediate_outputs

        market_state_shape, proportion_state_shape, complementary_state_shape = input_shapes
        self.num_symbols, self.num_properties, self.num_observations = market_state_shape
        self.num_complementary_data = complementary_state_shape


class BasicEvaluator(Evaluator):
    def __init__(
            self,
            input_shapes: List[Union[Sequence[int], int]],
            num_evaluator_outputs: int,
            num_extractor_outputs: int,
    ):
        super(BasicEvaluator, self).__init__(
            input_shapes=input_shapes,
            num_outputs=num_evaluator_outputs,
            num_intermediate_outputs=num_extractor_outputs,
        )

        self.extractor = nn.Linear(self.num_properties * self.num_observations, num_extractor_outputs)  # todo maybe do other classes 'extractors'
        self.linear = nn.Linear(self.num_symbols * num_extractor_outputs + self.num_symbols + 2, num_evaluator_outputs)

    def forward(
            self,
            market_state: torch.Tensor,
            proportions_state: torch.Tensor,
            complementary_state: torch.Tensor,
    ) -> torch.Tensor:  # todo need to find where the warning come from
        observation = torch.flatten(market_state, start_dim=2)
        features = apply_along_axis(self.extractor, observation)
        features_flattened = torch.flatten(features, start_dim=1)
        concatenated = torch.cat([features_flattened, complementary_state, proportions_state], dim=1)
        return self.linear(concatenated)
