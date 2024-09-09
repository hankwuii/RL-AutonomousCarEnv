import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    DQN Network
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int):
        """
        Args:
            state_dim (int): dimension of state
            action_dim (int): dimension of action
        """
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.sequential(x)

        return output
