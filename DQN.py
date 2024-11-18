import gym
import torch
import os

from copy import deepcopy
from memory import ReplayMemory
from qnets import QNetDiscrete
from utils import soft_update
from tqdm import tqdm

# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

#environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

#parameters
gamma = 0.98  # discount factor
tau = 0.05    # soft update rate
memory_limit = 100000

#Networks initialisation
Qnet = QNetDiscrete(state_dim, action_dim, learning_rate=1e-3).to(device)
target_Qnet = deepcopy(Qnet)
#Replay Buffer
memory = ReplayMemory(state_dim, 1, memory_limit)

def DQN_Step(batch_size=64):
    '''
    Single train step
    '''
    batch = memory.sample(batch_size)
    
    #---------------------Qnet training----------------------------------
    with torch.no_grad():
        next_Q_values = target_Qnet(batch.next_states).max(1)[0].unsqueeze(1)
        expected_Q_values = batch.rewards + gamma * next_Q_values * (1 - batch.dones)
    Qnet.train(batch.states, batch.actions, expected_Q_values)
    
    #----------------soft update target Qnet-------------------------------
    soft_update(target_Qnet, Qnet, tau)


def save(path='checkpoints'):
    '''Save model checkpoints'''
    if not os.path.exists(path): os.makedirs(path)
    torch.save({
        'Qnet_state_dict': Qnet.state_dict(),
        'target_Qnet_state_dict': target_Qnet.state_dict(),
    }, os.path.join(path, 'dqn_checkpoint.pth'))


def DQN_Train(iters=1000, batch_size=64, max_ep_len=1000,
              update_freq=1, eps=1, save_freq=1000):
    '''
    Train Qnets
    '''
    for _ in tqdm(range(iters)):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_ep_len:
            #Collect transition
            action = Qnet.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            
            #Add transition to memory
            memory.add_transition(state, action, next_state, reward, done)
            
            #if it's time to train
            if steps%update_freq == 0 and len(memory) > batch_size:
                for e in range(eps):
                    DQN_Step(batch_size)
                    
            #if it's time to save
            if steps%save_freq == 0:
                save()
                
            state = next_state
            steps += 1
        save()

DQN_Train()
