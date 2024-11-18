import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from utils import *


class BasePolicy(nn.Module):
    '''Base Policy Class'''
    def __init__(self):
        super().__init__()
        self.optimizer = None

    def forward(self, states):
        raise NotImplementedError('forward not implemented')

    def sample(self, states):
        dist = self.forward(states)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        log_probs = log_probs.unsqueeze(-1)
        return actions, log_probs

    def choose_action(self, state, training=False):
        with torch.no_grad():
            dist = self.forward(T(state))
            action = self._action_from_dist(dist, training)
        return np.round(action, 5)

    def _action_from_dist(self, dist, training=False):
        if training:
            action = dist.sample()
        else:
            action = dist.mean
        return action.cpu().squeeze(0).numpy()

    def train(self, loss, clip_value=1):
        optimize(self.parameters(), self.optimizer, loss)


class ContinuousPolicy(BasePolicy):
    '''Continuous action Policy'''
    def __init__(self, state_dim, action_dim,
                 learning_rate, std=None):
        super().__init__()
        self.std = std
        self.fcn = Sequential([state_dim, 128, 256, action_dim])

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.fcn(states)
        mean = torch.tanh(x)
        if self.std is None: return mean
        else: return Normal(mean, self.std)

    def _action_from_dist(self, dist, training=False):
        if self.std is None:
            action = dist
        else:
            if training:
                action = dist.sample()
            else:
                action = dist.mean
        return action.cpu().squeeze(0).numpy()


class DiscretePolicy(BasePolicy):
    '''Discrete action Policy'''
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.fcn = Sequential([state_dim, 128, 256, action_dim])

        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, states):
        x = self.fcn(states)
        probs = torch.softmax(x, dim=-1)
        return Categorical(probs)

    def _action_from_dist(self, dist, training=False):
        if training:
            action = dist.sample()
        else:
            action = dist.probs.argmax(dim=1)
        return action.item()


class SquashedGaussianPolicy(BasePolicy):
    '''Squashed Gaussian Policy'''
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
        return actions, log_probs

    def _action_from_dist(self, dist, training=False):
        sample = dist.sample() if training else dist.mean
        action = torch.tanh(sample).squeeze(0)
        return action.cpu().numpy()


# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import distrax

# class FlaxSquashedGaussianPolicy(nn.Module):
#     '''Squashed Gaussian Policy implemented in JAX/Flax using Distrax'''
#     action_dim: int
    
#     @nn.compact
#     def __call__(self, states, training=False):
#         x = nn.Dense(128)(states)
#         x = nn.relu(x)
#         x = nn.Dense(256)(x)
#         x = nn.relu(x)
        
#         mean = nn.Dense(self.action_dim)(x)
#         log_std = nn.Dense(self.action_dim)(x)
#         std = jnp.exp(jnp.clip(log_std, -20, 2))
        
#         base_dist = distrax.Normal(loc=mean, scale=std)
#         dist = distrax.Transformed(distribution=base_dist, bijector=distrax.Tanh())
        
#         if training:
#             samples, log_probs = dist.sample_and_log_prob(seed=self.make_rng('sampling'))
#         else:
#             samples = dist.mode()
#             log_probs = dist.log_prob(samples)
            
#         return samples, log_probs

#     def sample(self, states, rng):
#         actions, log_probs = self.apply({'params': self.params}, 
#                                       states, training=True, 
#                                       rngs={'sampling': rng})
#         return actions, log_probs

#     def choose_action(self, state, rng):
#         state = jnp.expand_dims(state, axis=0)
#         action, _ = self.apply({'params': self.params},
#                              state, training=False,
#                              rngs={'sampling': rng})
#         return jnp.squeeze(action, axis=0)

