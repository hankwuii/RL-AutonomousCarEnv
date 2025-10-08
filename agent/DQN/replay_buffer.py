# 儲存遊玩經驗供訓練使用
import numpy as np


class ReplayBuffer():
    def __init__(self,
                 memory_size: int,
                 state_dim: int):

        self.memory_size = memory_size
        self.obs_memory = np.zeros((self.memory_size, state_dim))
        self.next_obs_memory = np.zeros((self.memory_size, state_dim))
        self.action_memory = np.zeros((self.memory_size, 1))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.done_memory = np.zeros((self.memory_size, 1))

        self.mem_counter = 0

    def store_transition(self,
                         obs: np.ndarray,
                         action: int,
                         reward: float | int,
                         next_obs: np.ndarray,
                         done: bool) -> None:
        """
        Store transition in memory
        Args:
            obs (np.ndarray): current state
            action (int): action
            reward (float | int): reward
            next_obs (np.ndarray): next state
            done (bool): done
        """

        self.action_memory[self.mem_counter][0] = action
        self.obs_memory[self.mem_counter] = obs
        self.reward_memory[self.mem_counter][0] = reward
        self.next_obs_memory[self.mem_counter] = next_obs
        self.done_memory[self.mem_counter][0] = int(done)

        self.mem_counter += 1

        if self.mem_counter == self.memory_size:
            self.mem_counter = 0

    def sample_buffer(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly sample batch_size experiences from the memory

        Args:
            batch_size (int): batch size
        Returns:
            obs, next_obs, actions, rewards, done
        """
        max_mem = min(self.mem_counter, self.memory_size)
        batch_idx = np.random.choice(max_mem, batch_size)

        obs = self.obs_memory[batch_idx]
        next_obs = self.next_obs_memory[batch_idx]
        actions = self.action_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        done = self.done_memory[batch_idx]

        return obs, actions, rewards, next_obs, done
