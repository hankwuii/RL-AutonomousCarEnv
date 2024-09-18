import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.main(state)[:, 0]
