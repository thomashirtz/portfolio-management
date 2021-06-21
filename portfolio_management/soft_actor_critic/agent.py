import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa Remove 'Successfully opened dynamic library' tensorflow warning

from typing import Optional
from typing import Sequence
from typing import Union
from typing import Tuple

import numpy as np
from pathlib import Path
from pathlib import PurePath

import torch
import torch.optim as optim
import torch.nn.functional as F  # noqa : N812

from portfolio_management.soft_actor_critic.memory import ReplayBuffer
from portfolio_management.soft_actor_critic.critic import TwinnedQNetworks
from portfolio_management.soft_actor_critic.policy import StochasticPolicy

from portfolio_management.soft_actor_critic.utilities import eval_mode
from portfolio_management.soft_actor_critic.utilities import get_device
from portfolio_management.soft_actor_critic.utilities import save_model
from portfolio_management.soft_actor_critic.utilities import load_model
from portfolio_management.soft_actor_critic.utilities import update_network_parameters


class Agent:
    def __init__(self, observation_shape: int, num_actions: int, batch_size: int = 256, memory_size: int = 10e6,
                 learning_rate: float = 3e-4, alpha: float = 1, gamma: float = 0.99, tau: float = 0.005,
                 hidden_units: Optional[Sequence[int]] = None, load_models: bool = False,
                 checkpoint_directory: Optional[Union[Path, str]] = None):  # todo implement hard update

        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.num_actions = num_actions
        self.observation_shape = observation_shape

        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = ReplayBuffer(memory_size, self.observation_shape, self.num_actions)

        self.policy = StochasticPolicy(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)
        self.critic = TwinnedQNetworks(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)
        self.target_critic = TwinnedQNetworks(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_critic.eval()

        self.device = get_device()
        for network in [self.policy, self.critic, self.target_critic]:
            network.to(device=self.device)

        update_network_parameters(self.critic, self.target_critic, tau=1)

        self.alpha = torch.Tensor([alpha]).to(self.device)
        self.target_entropy = -torch.Tensor([num_actions]).to(self.device).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        self.policy_checkpoint_path = None
        self.critic_checkpoint_path = None

        if checkpoint_directory:
            if not isinstance(checkpoint_directory, PurePath):
                checkpoint_directory = Path(Path)
            checkpoint_directory.mkdir(parents=True, exist_ok=True)

            self.policy_checkpoint_path = checkpoint_directory / 'actor.pt'
            self.critic_checkpoint_path = checkpoint_directory / 'critic.pt'
            if load_models:
                load_model(self.policy, self.policy_checkpoint_path)
                load_model(self.critic, self.critic_checkpoint_path)

    def choose_action(self, observation, deterministically: bool = False) -> np.array:
        observation = torch.FloatTensor(observation).to(self.device)
        action = self.policy.act(observation, deterministic=deterministically)
        return action

    def remember(self, state, action, reward, new_state, done) -> None:
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self) -> dict:
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        critic_1_loss, critic_2_loss = self._critic_optimization(state, action, reward, next_state, done)
        policy_loss = self._policy_optimization(state)
        alpha_loss = self._entropy_optimization(state)

        update_network_parameters(self.critic, self.target_critic, self.tau)

        tensorboard_logs = {
            'loss/critic_1': critic_1_loss,
            'loss/critic_2': critic_2_loss,
            'loss/policy': policy_loss,
            'loss/entropy_loss': alpha_loss,
            'miscellaneous/alpha': self.alpha.clone().item(),
        }
        return tensorboard_logs

    def _critic_optimization(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                             next_state: torch.Tensor, done: torch.Tensor) -> Tuple[float, float]:
        with torch.no_grad():
            next_action, next_log_pi = self.policy.evaluate(next_state)
            next_q_target_1, next_q_target_2 = self.target_critic.forward(next_state, next_action)
            min_next_q_target = torch.min(next_q_target_1, next_q_target_2)
            next_q = reward + (1 - done) * self.gamma * (min_next_q_target - self.alpha * next_log_pi)

        q_1, q_2 = self.critic.forward(state, action)
        q_network_1_loss = F.mse_loss(q_1, next_q)
        q_network_2_loss = F.mse_loss(q_2, next_q)
        q_loss = (q_network_1_loss + q_network_2_loss) / 2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        return q_network_1_loss.item(), q_network_2_loss.item()

    def _policy_optimization(self, state: torch.Tensor) -> float:
        with eval_mode(self.critic):
            predicted_action, log_probabilities = self.policy.evaluate(state)
            q_1, q_2 = self.critic(state, predicted_action)
            min_q = torch.min(q_1, q_2)

            policy_loss = ((self.alpha * log_probabilities) - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            return policy_loss.item()

    def _entropy_optimization(self, state: torch.Tensor) -> float:
        with eval_mode(self.policy):
            _, log_pi = self.policy.evaluate(state)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            return alpha_loss.item()

    def save_models(self):
        save_model(self.policy, self.policy_checkpoint_path)
        save_model(self.critic, self.critic_checkpoint_path)

    def load_models(self):
        save_model(self.policy, self.policy_checkpoint_path)
        save_model(self.critic, self.critic_checkpoint_path)
