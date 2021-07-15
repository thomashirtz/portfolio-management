from typing import Tuple
from typing import Union
from typing import List
from typing import Sequence

import numpy as np


class ReplayBuffer:  # todo do more clean + random deletion + PER
    def __init__(
            self,
            max_size: int,
            observation_shapes: Sequence[Union[Sequence[int], int]],
            num_actions: int
    ):
        self.memory_counter = 0
        self.memory_size = int(max_size)

        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, num_actions), dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=np.float32)

        self.state_memories = [np.zeros((self.memory_size, *shape), dtype=np.float32) for shape in observation_shapes]
        self.new_state_memories = [np.zeros((self.memory_size, *shape), dtype=np.float32) for shape in observation_shapes]

    def store_transition(
            self,
            states: List[Union[list, np.array]],
            action,
            reward: float,
            new_states: List[Union[list, np.array]],
            done: Union[int, bool]
    ):
        index = self.memory_counter % self.memory_size

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        for i, state in enumerate(states):
            self.state_memories[i][index] = state
        for i, new_state in enumerate(new_states):
            self.new_state_memories[i][index] = new_state

        self.memory_counter += 1

    def sample_buffer(self, batch_size: int) -> Tuple[List[np.array], np.array, List[np.array], np.array, np.array]:
        max_mem = min(self.memory_counter, self.memory_size)

        batch = np.random.choice(max_mem, batch_size)

        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        done_batch = self.done_memory[batch]

        state_batch = [state_memory[batch] for state_memory in self.state_memories]
        new_state_batch = [state_memory[batch] for state_memory in self.new_state_memories]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch
