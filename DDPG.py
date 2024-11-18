import gym
import torch
import os

from copy import deepcopy
from memory import ReplayMemory
from qnets import QNetContinuous
from offline_policies import ContinuousPolicy
from utils import OUNoise, soft_update
from tqdm import tqdm
import numpy as np


# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

#environment
env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#parameters
gamma = 0.98  # discount factor
tau = 0.05    # soft update rate
lr_policy = 1e-3
lr_critic = 1e-3
memory_limit = 100000

#Networks initialisation
actor = ContinuousPolicy(state_dim, action_dim, lr_policy, std=None).to(device)
critic = QNetContinuous(state_dim, action_dim, lr_critic).to(device)
#target networks
target_actor = deepcopy(actor)
target_critic = deepcopy(critic)
#Replay Buffer
memory = ReplayMemory(state_dim, action_dim, memory_limit)
#Noise process
noise = OUNoise(mu=np.zeros(action_dim))


def DDPG_Step(batch_size=64):
    '''
    Single DDPG train step
    '''
    batch = memory.sample(batch_size)
    
    #---------------------critic training----------------------------------
    with torch.no_grad():
        next_actions = target_actor(batch.next_states)
        next_critic_values = target_critic(batch.next_states, next_actions)
        expected_critic_values = batch.rewards + gamma * next_critic_values * (1 - batch.dones)
    critic.train(batch.states, batch.actions, expected_critic_values)
    
    #---------------------actor training----------------------------------
    critic.requires_grad(False)
    actions = actor(batch.states)
    actor_loss = -critic(batch.states, actions).mean()
    actor.train(actor_loss)
    critic.requires_grad(True)
    
    #----------------soft update target networks---------------------------
    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)


def save(path='checkpoints'):
    '''Save model checkpoints'''
    if not os.path.exists(path): os.makedirs(path)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'target_actor_state_dict': target_actor.state_dict(),
        'target_critic_state_dict': target_critic.state_dict(),
    }, os.path.join(path, 'ddpg_checkpoint.pth'))


def DDPG_Train(iters=1000, batch_size=64, max_ep_len=1000,
               update_freq=1, eps=1, save_freq=1000):
    '''
    Train actor and critic networks
    '''
    for _ in tqdm(range(iters)):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_ep_len:
            #Collect transition
            action = actor.choose_action(state) + noise()
            next_state, reward, done, info, _ = env.step(action)
            
            #Add transition to memory
            memory.add_transition(state, action, next_state, reward, done)
            
            #if it's time to train
            if steps%update_freq == 0 and len(memory) > batch_size:
                for e in range(eps):
                    DDPG_Step(batch_size)
                    
            #if it's time to save
            if steps%save_freq == 0:
                save()
                
            state = next_state
            steps += 1
        save()

DDPG_Train()
