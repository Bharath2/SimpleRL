import numpy as np
from recordclass import recordclass

Transitions = recordclass('Transitions', ('states', 'actions', 'next_states', 'rewards', 'dones'))

class ReplayMemory():
    def __init__(self, state_dim, action_dim, maxlen = 1000):

        self.len = 0
        self.maxlen = maxlen

        self.states = np.zeros((maxlen, state_dim))
        self.actions = np.zeros((maxlen, action_dim))
        self.next_states = np.zeros((maxlen, state_dim))
        self.rewards = np.zeros((maxlen, 1))
        self.dones = np.zeros((maxlen, 1))


    def add_transition(self, state, action, next_state, reward, done):
        index = self.len % self.maxlen

        self.states[index] = state
        self.actions[index] = action
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done

        self.len += 1


    def sample(self, batch_size):
        maxind = min(self.len, self.maxlen)
        inds = np.random.choice(maxind, batch_size)
        batch = Transitions(self.states[inds],
                            self.actions[inds],
                            self.next_states[inds],
                            self.rewards[inds],
                            self.dones[inds])

        for i in range(5):
          batch[i] = torch.tensor(batch[i], dtype = torch.float)

        return batch

    def clear(self):
      self.len = 0

    def __len__(self):
      return min(self.len, self.maxlen)


class SequentialMemory():
    '''
    ToDo
    '''
    def __init__(self, state_dim, action_dim, maxlen = 1000):
        pass
