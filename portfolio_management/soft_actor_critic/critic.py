from typing import List
from typing import Tuple
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn

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
    def __init__(self, input_dims: int, num_actions: int, hidden_units: Optional[Sequence[int]] = None):
        super(TwinnedQNetworks, self).__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.q_network_1 = QNetwork(input_dims=input_dims, num_actions=num_actions, hidden_units=self.hidden_units)
        self.q_network_2 = QNetwork(input_dims=input_dims, num_actions=num_actions, hidden_units=self.hidden_units)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        q_1 = self.q_network_1(state, action)
        q_2 = self.q_network_2(state, action)
        return q_1, q_2
