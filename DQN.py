import gym
import torch

from copy import deepcopy
from .qnets import QNetDiscrete
from .memory import ReplayMemory

# if gpu is to be used
if torch.cuda.is_available():
  device = torch.device("cuda")
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
  device = torch.device("cpu")
  torch.set_default_tensor_type(torch.FloatTensor)

#environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space
action_dim = env.action_space

#custom feature extractor like CNN can be added (look at source code)
Qnet = QNetDiscrete(state_dim, action_dim, learning_rate = 1e-3)
target_Qnet = deepcopy(Qnet)

def DQN_Step(batch_size = 64):
    '''
    Single train step
    '''
    batch = memory.sample(batch_size)
   #---------------------Qnet training----------------------------------
    with torch.no_grad():
      next_Q_values = target_critic(batch.next_states).max(1)[0].unsqueeze(1)
      expected_Q_values = batch.rewards + gamma * next_Q_values * (1 - batch.dones)
    Qnet.train(batch.states, batch.actions, expected_Q_values)
   #----------------soft update target Qnet-------------------------------
    soft_update(target_Qnet, Qnet, tau)
   #-----------------------------------------------------------------------


def DQN_Train(iters = 1000, batch_size = 64, max_ep_len = 1e5,
                update_freq = 1, eps = 1, save_freq = 1000):
    '''
    Train Qnets
    '''
    for _ in tqdm(range(iters)):
        state = env.reset()
        done, steps = 0, 0
        while not done and steps < maxsteps:
          #Collect transition
          action = Qnet.choose_action(state)
          next_state, reward, done, info = env.step(action)
          #Add transition to memory
          memory.add_transition(state, action, next_state, reward, done)
          #if it's time to train
          if steps%update_freq == 0 and len(memory) > 1000:
            for e in range(eps): DQN_Step(batch_size)
          #if it's time to save
          if steps%save_freq == 0: save()
          state = next_state
          steps += 1

DQN_Train()
