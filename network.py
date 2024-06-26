import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, hidden_layers, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='models'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')
        self.hidden_layers = hidden_layers

        if self.hidden_layers == 1:
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

            # Divergence from the paper
            self.bn1 = nn.LayerNorm(self.fc1_dims)

            self.action_value = nn.Linear(self.n_actions, self.fc1_dims)
            self.q = nn.Linear(self.fc1_dims, 1)
        elif self.hidden_layers == 2:
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

            #Divergence from the paper
            self.bn1 = nn.LayerNorm(self.fc1_dims)
            self.bn2 = nn.LayerNorm(self.fc2_dims)

            self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
            self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device('cuda:0')
        # self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):

        if self.hidden_layers == 1:
            state_value = self.fc1(state)
            state_value = self.bn1(state_value)
            state_value = F.relu(state_value)
            action_value = self.action_value(action)
            state_action_value = F.relu(T.add(state_value, action_value))
            state_action_value = self.q(state_action_value)


        elif self.hidden_layers == 2:
            state_value = self.fc1(state)
            state_value = self.bn1(state_value)
            state_value = F.relu(state_value)
            state_value = self.fc2(state_value)
            state_value = self.bn2(state_value)
            action_value = self.action_value(action)
            state_action_value = F.relu(T.add(state_value, action_value))
            state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print( '... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print( '... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))




class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, hidden_layers, fc1_dims, fc2_dims, action_interval,
                 n_actions, name, chkpt_dir='models'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')
        self.hidden_layers = hidden_layers
        self.action_interval = action_interval

        if self.hidden_layers == 1:
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

            self.bn1 = nn.LayerNorm(self.fc1_dims)

            # batch norm: self.bn1 = nn.BatchNorm1d(self.fc1_dims), self.bn2 = nn.BatchNorm1d(self.fc2_dims)

            self.mu = nn.Linear(self.fc1_dims, self.n_actions)

        elif self.hidden_layers == 2:
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

            self.bn1 = nn.LayerNorm(self.fc1_dims)
            self.bn2 = nn.LayerNorm(self.fc2_dims)

            #batch norm: self.bn1 = nn.BatchNorm1d(self.fc1_dims), self.bn2 = nn.BatchNorm1d(self.fc2_dims)

            self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0')
        # self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        if self.hidden_layers == 1:
            x = self.fc1(state)
            x = self.bn1(x)
            x = F.relu(x)
            x = T.tanh(self.mu(
                x)) * self.action_interval  # if the action range is not -1, +1 then you can change the boundaries here, like *2 --> [-2, +2], etc.

        elif self.hidden_layers == 2:
            x = self.fc1(state)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = T.tanh(self.mu(x))*self.action_interval   #if the action range is not -1, +1 then you can change the boundaries here, like *2 --> [-2, +2], etc.

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))






































































