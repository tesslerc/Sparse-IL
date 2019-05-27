import torch
import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'success', 'timeout'))


class ReplayBuffer:
    def __init__(self, capacity, hist_len=1, state_parser=lambda x: x):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.hist_len = hist_len
        self.state_parser = state_parser

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []

        mem_size = len(self.memory)
        for _ in range(batch_size):
            idx = random.randint(0, mem_size - 2)
            while self.memory[idx].timeout:
                idx = random.randint(0, mem_size - 2)
            transition = self.memory[idx]
            batch_state.append(self.state_parser(self._build_state(self.memory, idx)))
            batch_action.append(transition.action)
            batch_reward.append(transition.reward)
            batch_done.append(transition.done)
            batch_next_state.append(self.state_parser(self._build_state(self.memory, idx + 1)))
        return torch.stack(batch_state), batch_action, batch_reward, torch.stack(batch_next_state), batch_done

    def _build_state(self, memory, index):
        state = [memory[index].state]
        for hist in range(self.hist_len - 1):
            idx = (index - hist - 1) % self.capacity
            if idx > len(memory):
                break
            if memory[idx].done:
                break

            state.insert(0, memory[idx].state)
        for _ in range(self.hist_len - len(state)):
            state.insert(0, None)
        return state

    def __len__(self):
        return len(self.memory)


class SuccessReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_parser=lambda x: x, success_sample_prob=0.2, success_relative_capacity=0.1,
                 max_trajectory_length=200, hist_len=1):
        super(SuccessReplayBuffer, self).__init__(capacity=capacity, state_parser=state_parser, hist_len=hist_len)
        self.success_sample_prob = success_sample_prob
        self.success_capacity = int(self.capacity * success_relative_capacity)
        self.success_memory = []
        self.success_position = 0
        self.max_trajectory_length = max_trajectory_length

    def add(self, *args):
        position = self.position
        super(SuccessReplayBuffer, self).add(*args)
        if self.memory[position].success:
            """
            If trajectory lead to a successful finish of the task, find the start index of the trajectory and add while
            trajectory to success replay memory.
            """
            start_idx = position
            num_backtracked = 0
            if len(self.memory) < self.capacity:
                while start_idx > 1 and num_backtracked <= self.max_trajectory_length:
                    start_idx = start_idx - 1
                    num_backtracked += 1
            else:
                while not self.memory[(start_idx - 1) % self.capacity].done and num_backtracked <= self.max_trajectory_length:
                    start_idx = (start_idx - 1) % self.success_capacity
                    num_backtracked += 1

            while True:
                if len(self.success_memory) < self.success_capacity:
                    self.success_memory.append(None)
                self.success_memory[self.success_position] = self.memory[start_idx]
                self.success_position = (self.success_position + 1) % self.success_capacity
                start_idx = (start_idx + 1) % self.capacity
                if start_idx == position + 1:
                    return

    def sample(self, batch_size):
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []

        mem_size = len(self.memory)
        success_mem_size = len(self.success_memory)
        for _ in range(batch_size):
            if random.random() < self.success_sample_prob and success_mem_size > batch_size:
                idx = random.randint(0, success_mem_size - 2)
                transition = self.success_memory[idx]
                batch_state.append(self.state_parser(self._build_state(self.success_memory, idx)))
                batch_action.append(transition.action)
                batch_reward.append(transition.reward)
                batch_done.append(transition.done)
                batch_next_state.append(self.state_parser(self._build_state(self.success_memory, idx + 1)))
            else:
                idx = random.randint(0, mem_size - 2)
                while self.memory[idx].timeout:
                    idx = random.randint(0, mem_size - 2)
                transition = self.memory[idx]
                batch_state.append(self.state_parser(self._build_state(self.memory, idx)))
                batch_action.append(transition.action)
                batch_reward.append(transition.reward)
                batch_done.append(transition.done)
                batch_next_state.append(self.state_parser(self._build_state(self.memory, idx + 1)))
        return torch.stack(batch_state), batch_action, batch_reward, torch.stack(batch_next_state), batch_done
