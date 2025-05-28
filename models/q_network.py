# models/q_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):  # Simple FC
    def __init__(self, state_dim=4, action_dim=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
