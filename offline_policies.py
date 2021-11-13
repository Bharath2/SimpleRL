import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from utils import *


class BasePolicy(nn.Module):
    '''Base Policy Class'''
    def __init__(self):
        super().__init__()
        self.optimizer = None

    def forward(self, states):
        raise NotImplemetedError('forward not implemented')

    def sample(self, states):
        dist = self.forward(states)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs.mean(axis = 1)

    def choose_action(self, state, training = False):
        with torch.no_grad():
          dist = self.forward(T(state))
          action = self._action_from_dist(dist, training)
        return np.round(action, 5)

    def _action_from_dist(self, dist, training = False):
        if training: action = dist.sample()
        else: action = dist.mean
        return action.cpu().squeeze(0).numpy()

    def train(self, loss, clip_value = 1):
        optimize(self.parameters(), self.optimizer, loss)


class ContinuousPolicy(BasePolicy):
    '''  Continuous action Policy  '''
    def __init__(self, state_dim, action_dim,
                       learning_rate, std = None):
        super().__init__()
        self.std = std
        self.fcn = Sequential([state_dim, 128, 256, action_dim])

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.fcn(states)
        mean = torch.tanh(x)
        if self.std is None: return mean
        else: return Normal(mean, self.std)

    def _action_from_dist(self, dist, training = False):
        if self.std is None:
            action = dist
        else:
            if training: action = dist.sample()
            else: action = dist.mean
        return action.cpu().squeeze(0).numpy()


class DiscretePolicy(BasePolicy):
    '''  Discrete action Policy  '''
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.fcn = Sequential([state_dim, 128, 256, action_dim])

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.fcn(states)
        probs = torch.softmax(x)
        return Categorical(probs)

    def _action_from_dist(self, dist, training = False):
        if training:
            action = dist.sample()
        else:
            action = dist.probs.argmax(1)
        return action.item()


class SquashedGaussianPolicy(BasePolicy):
    '''  Squashed Gaussian Policy  '''
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.fcn = Sequential([state_dim, 128, 256])
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.fcn(states)
        x = torch.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std.clamp(-20, 2))
        return Normal(mean, std)

    def sample(self, states):
        dist = self.forward(states)
        samples = dist.rsample()
        actions = torch.tanh(samples)
        log_probs = dist.log_prob(samples) - torch.log(1 - actions**2 + 1e-6)
        return actions, log_probs.mean(axis = 1)

    def _action_from_dist(self, dist, training = False):
        sample = dist.sample() if training else dist.mean
        action = torch.tanh(sample).squeeze(0)
        return action.cpu().numpy()
