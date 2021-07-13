from typing import List
from typing import Tuple
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn

from copy import deepcopy

from portfolio_management.soft_actor_critic.evaluators import Evaluator
from portfolio_management.soft_actor_critic.utilities import weight_initialization
from portfolio_management.soft_actor_critic.utilities import get_multilayer_perceptron


class QNetwork(nn.Module):
    """ Q Network module """
    def __init__(self, input_dims: int, num_actions: int, hidden_units: List[int]):
        super(QNetwork, self).__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = hidden_units

        units = [input_dims + num_actions] + hidden_units + [1]
        self.multilayer_perceptron = get_multilayer_perceptron(units, keep_last_relu=False)

        self.apply(weight_initialization)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.cat([state, action], dim=1)
        q_value = self.multilayer_perceptron(tensor)
        return q_value


class TwinnedQNetworks(nn.Module):
    """ Class containing two Q Networks """
    def __init__(
            self,
            evaluator: Evaluator,
            num_actions: int,
            hidden_units: Optional[Sequence[int]] = None
    ):
        super(TwinnedQNetworks, self).__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = evaluator.num_evaluator_outputs
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.evaluator_1 = deepcopy(evaluator)
        self.evaluator_2 = deepcopy(evaluator)

        self.q_network_1 = QNetwork(input_dims=self.input_dims, num_actions=num_actions, hidden_units=self.hidden_units)
        self.q_network_2 = QNetwork(input_dims=self.input_dims, num_actions=num_actions, hidden_units=self.hidden_units)

    def forward(
            self,
            tensor_list: List[torch.Tensor],
            action  # todo put type
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # todo maybe pass into the module before splitting
        input_1 = self.evaluator_1(*tensor_list)
        input_2 = self.evaluator_2(*tensor_list)
        q_1 = self.q_network_1(input_1, action)
        q_2 = self.q_network_2(input_2, action)
        return q_1, q_2
