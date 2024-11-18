import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from utils import *

class QNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(action_dim, 32)
        self.fcn = Sequential([64 + 32, 256, 1])

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states, actions):
        x1 = torch.relu(self.fc1(states))
        x2 = torch.relu(self.fc2(actions))
        x = torch.cat((x1, x2), 1)
        return self.fcn(x)

    def train(self, states, actions, expected_values):
        q_values = self.forward(states, actions)
        loss = F.smooth_l1_loss(expected_values, q_values)
        optimize(self.parameters(), self.optimizer, loss)


class DQNContinuous(nn.Module):
    '''     Double Q-Network Critic    '''
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.q1 = QNetContinuous(state_dim, action_dim, learning_rate)
        self.q2 = QNetContinuous(state_dim, action_dim, learning_rate)

    def forward(self, states, actions):
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        return torch.min(q1, q2)

    def train(self, states, actions, expected_values):
        self.q1.train(states, actions, expected_values)
        self.q2.train(states, actions, expected_values)

    def requires_grad(self, boolean = True):
        for param in self.parameters():
            param.requires_grad = boolean

# import jax
# import jax.numpy as jnp
# import flax.linen as nn

# class FlaxQNetContinuous(nn.Module):
#     '''Single Q-Network implemented in JAX/Flax'''
#     action_dim: int
    
#     @nn.compact
#     def __call__(self, states, actions):
#         x1 = nn.Dense(64)(states)
#         x1 = nn.relu(x1)
        
#         x2 = nn.Dense(32)(actions) 
#         x2 = nn.relu(x2)
        
#         x = jnp.concatenate([x1, x2], axis=-1)
#         x = nn.Dense(256)(x)
#         x = nn.relu(x)
#         x = nn.Dense(1)(x)
#         return x

# class FlaxDQNContinuous(nn.Module):
#     '''Double Q-Network Critic implemented in JAX/Flax'''
#     action_dim: int
    
#     def setup(self):
#         self.q1 = FlaxQNetContinuous(self.action_dim)
#         self.q2 = FlaxQNetContinuous(self.action_dim)
    
#     def __call__(self, states, actions):
#         q1 = self.q1(states, actions)
#         q2 = self.q2(states, actions)
#         return jnp.minimum(q1, q2)
    
#     def train_step(self, params, states, actions, expected_values, optimizer):
#         def loss_fn(params):
#             q1 = self.q1.apply({'params': params['q1']}, states, actions)
#             q2 = self.q2.apply({'params': params['q2']}, states, actions)
#             loss1 = jnp.mean((q1 - expected_values) ** 2)
#             loss2 = jnp.mean((q2 - expected_values) ** 2)
#             return loss1 + loss2
            
#         grad_fn = jax.value_and_grad(loss_fn)
#         loss, grads = grad_fn(params)
#         updates, optimizer = optimizer.update(grads, optimizer)
#         params = optax.apply_updates(params, updates)
#         return params, optimizer, loss


class QNetDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, extractor = None):
        super().__init__()
        self.action_dim = action_dim
        # linear feature extractor
        if extractor is None:
            self.extract = nn.Linear(state_dim, 128)
            self.hidden_dim = 128
        # if specified, like cnn
        else:
            self.extract = extractor(state_dim)
            self.hidden_dim = self.extract.out_dim

        self.fcn = Sequential([self.hidden_dim, 256, action_dim])
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.extract(states)
        x = self.fcn(x)
        return x

    def train(self, states, actions, expected_values):
        Q_out = self.forward(states)
        Q_values = Q_out.gather(1, actions)
        loss = F.smooth_l1_loss(expected_values, Q_values)
        optimize(self.parameters(), self.optimizer, loss)

    def choose_action(self, state, epsilon = 0.2):
        if np.random.uniform() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
              dist = self.forward(T(state))
              action = dist.probs.argmax(1)
            return np.round(action.item(), 5)


class DQNDiscrete(nn.Module):
    '''     Double Q-Network Critic    '''
    def __init__(self, state_dim, action_dim, learning_rate, extractor = None):
        super().__init__()
        self.action_dim = action_dim
        self.q1 = QNetDiscrete(state_dim, action_dim, learning_rate, extractor)
        self.q2 = QNetDiscrete(state_dim, action_dim, learning_rate, extractor)

    def forward(self, states):
        q1 = self.q1(states)
        q2 = self.q2(states)
        return torch.min(q1, q2, axis = 1)

    def train(self, states, actions, expected_values):
        self.q1.train(states, actions, expected_values)
        self.q2.train(states, actions, expected_values)

    def requires_grad(self, boolean = True):
        for param in self.parameters():
            param.requires_grad = boolean

    def choose_action(self, state, epsilon = 0.2):
        if np.random.uniform() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
              dist = self.forward(T(state))
              action = dist.probs.argmax(1)
            return np.round(action.item(), 5)
