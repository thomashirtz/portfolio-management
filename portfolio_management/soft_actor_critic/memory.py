import numpy as np

from typing import Tuple
from typing import Union
from typing import Sequence


class ReplayBuffer:  # todo maybe do more clean in the future
    def __init__(self, max_size: int, input_shape: Union[Sequence[int], int], num_actions: int):
        self.memory_counter = 0
        self.memory_size = int(max_size)

        self.state_memory = np.zeros((self.memory_size, input_shape), dtype=np.float32)  # todo rename
        self.action_memory = np.zeros((self.memory_size, num_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, input_shape), dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, state: Union[list, np.array], action, reward: float, new_state: Union[list, np.array], done: Union[int, bool]):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size: int) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        max_mem = min(self.memory_counter, self.memory_size)

        batch = np.random.choice(max_mem, batch_size)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        done_batch = self.done_memory[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch
