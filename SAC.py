import gym
import torch
import os

from copy import deepcopy
from memory import ReplayMemory
from qnets import DQNContinuous
from offline_policies import SquashedGaussianPolicy

from utils import *
from tqdm import tqdm

# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)


print(device)
#environment
env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#parameters
gamma = 0.98  # discount factor
alpha = 0.1   # entropy reward coeff
tau = 0.95      # soft update rate
lr_policy = 1e-3
lr_critic = 1e-3
memory_limit = 100000

#Networks initialisation
actor  = SquashedGaussianPolicy(state_dim, action_dim, lr_policy).to(device)
critic = DQNContinuous(state_dim, action_dim, lr_critic).to(device)
target_critic = deepcopy(critic)
#Replay Buffer
memory = ReplayMemory(state_dim, action_dim, memory_limit)


def SAC_Step(batch_size = 64):
    '''
    Single sac train step
    '''
    batch = memory.sample(batch_size)
    
   #---------------------critic training----------------------------------
    with torch.no_grad():
      next_actions, log_probs = actor.sample(batch.next_states)
      next_critic_values = target_critic(batch.next_states, next_actions)
      expected_critic_values = batch.rewards + gamma * (next_critic_values - alpha*log_probs) * (1 - batch.dones)
    critic.train(batch.states, batch.actions, expected_critic_values)
    
   #---------------------actor training----------------------------------
    critic.requires_grad(False)
    actions, log_probs = actor.sample(batch.states)
    critic_value = critic(batch.states, actions)
    actor_loss = -torch.mean(critic_value - alpha*log_probs)
    actor.train(actor_loss)
    critic.requires_grad(True)
    
   #----------------soft update target critic-------------------------------
    for target_param, source_param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
   #------------------------------------------------------------------------


def save(path='checkpoints'):
    '''Save model checkpoints'''
    if not os.path.exists(path): os.makedirs(path)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'target_critic_state_dict': target_critic.state_dict(),
    }, os.path.join(path, 'sac_checkpoint.pth'))


def SAC_Train(iters = 2, batch_size = 64, max_ep_len = 1e5,
                update_freq = 1, eps = 1, save_freq = 1000):
    '''
    Train actor and critic networks
    '''
    for _ in tqdm(range(iters)):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_ep_len:
          #Collect transition
          action = actor.choose_action(state)
          next_state, reward, done, info, _ = env.step(action)
          
          #Add transition to memory
          memory.add_transition(state, action, next_state, reward, done)
          #if it's time to train
          if steps%update_freq == 0 and len(memory) > 1000:
            for e in range(eps): 
              SAC_Step(batch_size)
          #save at regular intervals and at the end of episode
          if steps%save_freq == 0:
            save()
          state = next_state
          steps += 1
    save()

SAC_Train()
