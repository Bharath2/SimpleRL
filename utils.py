import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_


def T(x):
    '''
    From list or numpy array to tensor
    '''
    return torch.as_tensor(x).unsqueeze(0)


def Sequential(dims, activation = nn.ReLU):
    '''
    Returns sequential model from specified by the dims list
    '''
    if len(dims) < 2: return None

    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers.append(activation())
        layers.append(nn.Linear(dims[i], dims[i + 1]))

    return nn.sequential(*layers)


class OUNoise:
    def __init__(self, mu):
        self.theta = 0.1
        self.dt = 0.01
        self.sigma = 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def optimize(parameters, optimizer, loss, clip_value = 1):
    '''
    Single optimize step
    '''
    optimizer.zero_grad()
    loss.backward()
    clip_grad_value_(parameters, clip_value)
    optimizer.step()


def soft_update(target, source, tau):
	for a, b in zip(target.parameters(), source.parameters()):
		a.data.copy_(a.data * (1.0 - tau) + b.data * tau)
