import os

import torch
import torch.nn as nn
import numpy as np
from torch.optim.adam import Adam

from .policy_network import PolicyNet
from .value_network import ValueNet


class PPOAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_step: int,
                 gamma: float = 0.99,
                 lamb: float = 0.95,
                 lr: float = 1e-4,
                 clip_val: float = 0.2,
                 max_grad_norm: float = 0.5,
                 ent_weight: float = 0.01,
                 sample_n_epoch: int = 4,
                 sample_mb_size: int = 64,
                 is_training: bool = True,
                 model_path: str = "./agent/weight.pth",
                 device: str = "cuda:0"
                 ) -> None:
        # Initialize hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_step = max_step
        self.gamma = gamma
        self.lamb = lamb
        self.lr = lr
        self.clip_val = clip_val
        self.max_grad_norm = max_grad_norm
        self.ent_weight = ent_weight
        self.sample_n_epoch = sample_n_epoch
        self.sample_mb_size = sample_mb_size
        self.is_training = is_training
        self.model_path = model_path
        self.device = device

        # Build networks
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.value_net = ValueNet(self.state_dim).to(self.device)
        self.opt_policy = torch.optim.Adam(self.policy_net.parameters(), lr)
        self.opt_value = torch.optim.Adam(self.value_net.parameters(), lr)

        # Storage trajectory
        self.mb_obs = np.zeros((self.max_step, self.state_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, self.action_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)

        self.load_model()

    def compute_discounted_return(self, rewards, last_value):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t + 1]

        return returns

    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lamb * last_gae_lam

        return advs + values

    def get_action(self, obs: dict):
        """
            Get action based on observation

            Args:
                obs: dict
                    `{'rgb_image': ndarray(128, 128, 3), 'lidar': ndarray(1080,), 'pose': ndarray(6,), 'velocity': ndarray(6,), 'acceleration': ndarray(6,), time: ndarray(1,}`

            Returns: dict
                `{'motor': float,"steering": float}`
        """
        # TODO: Select action
        state = obs['lidar'].copy()  # (1080,)
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)

        # Normalize state
        state = (state - state.mean()) / (state.std() + 1e-8)

        if self.is_training:
            action, a_logp = self.policy_net(state)
            value = self.value_net(state)

            action = action.cpu().numpy()
            a_logp = a_logp.cpu().numpy()
            value = value.cpu().numpy()

            return action, a_logp, value
        else:
            pass

    def get_last_value(self, obs: dict):
        state = obs['lidar'].copy()  # (1080,)
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)

        # Normalize state
        state = (state - state.mean()) / (state.std() + 1e-8)

        return self.value_net(state).cpu().numpy()

    def learn(self,
              mb_states, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
        mb_states = torch.from_numpy(mb_states).to(self.device)
        mb_actions = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values = torch.from_numpy(mb_old_values).to(self.device)
        mb_advs = torch.from_numpy(mb_advs).to(self.device)
        mb_returns = torch.from_numpy(mb_returns).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)
        episode_length = len(mb_states)
        rand_idx = np.arange(episode_length)
        sample_n_mb = episode_length // self.sample_mb_size

        if sample_n_mb <= 0:
            sample_mb_size = episode_length
            sample_n_mb = 1
        else:
            sample_mb_size = self.sample_mb_size

        for i in range(self.sample_n_epoch):
            np.random.shuffle(rand_idx)

            for j in range(sample_n_mb):
                # Randomly sample a batch for training
                sample_idx = rand_idx[j * sample_mb_size: (j + 1) * sample_mb_size]
                sample_states = mb_states[sample_idx]
                sample_actions = mb_actions[sample_idx]
                sample_old_values = mb_old_values[sample_idx]
                sample_advs = mb_advs[sample_idx]
                sample_returns = mb_returns[sample_idx]
                sample_old_a_logps = mb_old_a_logps[sample_idx]

                sample_a_logps, sample_ents = self.policy_net.evaluate(sample_states, sample_actions)
                sample_values = self.value_net(sample_states)
                ent = sample_ents.mean()

                # Compute value loss
                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val,
                                                              self.clip_val)
                v_loss1 = (sample_returns - sample_values) ** 2
                v_loss2 = (sample_returns - v_pred_clip) ** 2
                v_loss = torch.max(v_loss1, v_loss2).mean()

                # Compute value loss
                ratio = (sample_a_logps - sample_old_a_logps).exp()
                pg_loss1 = ratio * -sample_advs
                pg_loss2 = torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val) * -sample_advs
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight * ent

                # Train actor
                self.opt_policy.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                # Train critic
                self.opt_value.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.opt_value.step()

    def store_trajectory(self,
                         obs: dict,
                         action: dict[str, float],
                         value,
                         reward: float | int,
                         a_logp,
                         step: int) -> None:

        obs = obs['lidar'].copy()

        # Normalize state
        state = (obs - obs.mean()) / (obs.std() + 1e-8)

        self.mb_obs[step] = state
        self.mb_actions[step] = action
        self.mb_values[step] = value
        self.mb_rewards[step] = reward
        self.mb_a_logps[step] = a_logp

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path) is False:
            print(f"Model path {self.model_path} does not exist")
            return

        print(f"Load model from {self.model_path}")
        self.policy_net.load_state_dict(torch.load(self.model_path))
