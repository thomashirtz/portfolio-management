import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, num_observations, num_properties: int = 9):
        super(BasicModel, self).__init__()
        self.num_properties = num_properties
        self.num_observations = num_observations
        self.linear = nn.Linear(num_properties*num_observations, num_properties)

    def forward(self, observation: torch.Tensor):
        observation = torch.flatten(observation, start_dim=1)
        return self.linear(observation)
