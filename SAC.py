import gym
import torch

from copy import deepcopy
from .memory import ReplayMemory
from .qnets import DQNContinuous
from .offline_policies import SquashedGaussianPolicy

# if gpu is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

#environment
env = gym.make('CartPoleContinuous-v1')
state_dim = env.observation_space
action_dim = env.action_space

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
    soft_update(target_critic, critic, tau)
   #------------------------------------------------------------------------


def SAC_Train(iters = 1000, batch_size = 64, max_ep_len = 1e5,
                update_freq = 1, eps = 1, save_freq = 1000):
    '''
    Train actor and critic networks
    '''
    for _ in tqdm(range(iters)):
        state = env.reset()
        done, steps = 0, 0
        while not done and steps < maxsteps:
          #Collect transition
          action = actor.choose_action(state)
          next_state, reward, done, info = env.step(action)
          #Add transition to memory
          memory.add_transition(state, action, next_state, reward, done)
          #if it's time to train
          if steps%update_freq == 0 and len(memory) > 1000:
            for e in range(eps): SAC_Step(batch_size)
          #if it's time to save
          if steps%save_freq == 0: save()
          state = next_state
          steps += 1

SAC_Train()
