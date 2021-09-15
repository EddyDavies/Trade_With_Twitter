import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, units=32):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dims, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, units=32):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dims, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
