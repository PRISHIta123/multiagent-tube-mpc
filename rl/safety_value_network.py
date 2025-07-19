# safety_value_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SafetyValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(SafetyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return value
