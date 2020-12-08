import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from nik_utils import make_env, Storage, orthogonal_init
from models_ppo import Flatten, Nature_Encoder, xavier_uniform_init
from models_ppo import ResidualBlock, Empala_Encoder, Policy, Empala_Encoder_One
from data_aug import RandGray, random_color_jitter, random_cutout
from time import time
from TransformLayer import ColorJitterLayer
from torch.distributions import Beta


# Hyperparameters
inp = sys.argv[1:] # 1 = model_name, 2 = encoder_index, 3 = aug_index, 4 = mix_reg, 5 = game
model_name = inp[0]
total_steps = 8e6
num_envs = 32
num_levels = 100
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
encoder_type_indnex = ["Impala","Nature","Impala_One"]
data_aug_index = ['none','grayscale','random_cutout','color_jitter']
encoder_type = encoder_type_indnex[int(inp[1])] # 0 = Impala, 1 = Nature
data_aug = data_aug_index[int(inp[2])]
tof = [False,True]
do_mixreg = tof[int(inp[3])]
game = inp[4]
beta_dist = Beta(torch.FloatTensor([0.2]), torch.FloatTensor([0.2]))

print("Run name",model_name)
print("Encoder used",encoder_type)
print("Augumentation used",data_aug)
print("Mixreg?" ,do_mixreg)
print("Game",game)

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels,env_name=game)
observation_space = env.observation_space
action_space = env.action_space.n
print('Observation space:', observation_space)
print('Action space:', action_space)
in_channels = observation_space.shape[0]
num_actions = action_space
feature_dim = 512 
nenv = env.num_envs

# Define network
if (encoder_type == "Impala"): 
  print("Beware of the Impala!")
  encoder = Empala_Encoder(in_channels, feature_dim)
elif (encoder_type == "Nature"): 
  print("Embrace Nature!")
  encoder = Nature_Encoder(in_channels, feature_dim)
elif (encoder_type == "Impala_One"):
  print("Small but good")
  encoder = Empala_Encoder_One(in_channels,feature_dim)
  
# Initialize the policy
policy = Policy(encoder, feature_dim, num_actions) # added
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
device = torch.device('cpu')

# Change the first observations to desired augmentation
if data_aug == 'grayscale':
  obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
  obs[:] = env.reset()

  # Do the grayscale and transfer to tensor
  augs_funcs = RandGray(batch_size=num_envs, p_rand=0.4) # added
  obs = augs_funcs.do_augmentation(obs)
  obs = torch.from_numpy(obs)
elif data_aug == 'random_cutout':
  # Initialize as a numpy array then convert to tensor
  obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
  obs[:] = env.reset()

  # Do the cutout and transfer to tensor
  obs = random_cutout(obs,12,24)
  obs = torch.from_numpy(obs)
elif data_aug == 'color_jitter':
  obs = random_color_jitter(obs,p=0.4)


# There is a mismatch from the repo, there the observations are initialized as numpy array
# nenv = env.num_envs # added
# obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
# obs[:] = env.reset()

step = 0
# Initilize mean_reward for each setup that we store in the end
mean_rewards = []
mean_rewards_done = []
first_loop = True
start = time() # Lets measure how long each training task takes
while step < total_steps:
  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

    # numpy obs
    next_obs[:], reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    # obs = next_obs

    # Make augmented transformation, probably possible to do this another way, like in a class to avoid the if statements
    if data_aug == 'grayscale':
      obs = torch.from_numpy(next_obs)
      obs = augs_funcs.do_augmentation(obs)
    elif data_aug == 'random_cutout':
      obs = random_cutout(next_obs,12,24) # change to 6,18 in future runs.
      obs = torch.from_numpy(obs)
    elif data_aug == 'color_jitter':
      obs = torch.from_numpy(next_obs)
      obs = random_color_jitter(obs,p=0.4)
    else:
      obs = torch.from_numpy(next_obs)

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      if do_mixreg:
          b_obs_c, b_action_c, b_log_prob_c, b_value_c, b_returns_c, b_advantage_c = batch
          lambda_mix = beta_dist.sample((batch_size,)).cuda()
          index_ij = torch.randint(0, batch_size-1, (batch_size,2))
          for ii,jj in enumerate(index_ij):
            b_obs_c[ii] = lambda_mix[ii]*b_obs[jj[0]] + (1-lambda_mix[ii])*b_obs[jj[1]]
            b_log_prob_c[ii] = lambda_mix[ii]*b_log_prob[jj[0]] + (1-lambda_mix[ii])*b_log_prob[jj[1]]
            b_value_c[ii] = lambda_mix[ii]*b_value[jj[0]] + (1-lambda_mix[ii])*b_value[jj[1]]
            b_returns_c[ii] = lambda_mix[ii]*b_returns[jj[0]] + (1-lambda_mix[ii])*b_returns[jj[1]]
            b_advantage_c[ii] = lambda_mix[ii]*b_advantage[jj[0]] + (1-lambda_mix[ii])*b_advantage[jj[1]]
            if (lambda_mix[ii] >= 0.5):
              b_action_c[ii] = b_action[jj[0]]
            else:
              b_action_c[ii] = b_action[jj[1]]
          b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = b_obs_c, b_action_c, b_log_prob_c, b_value_c, b_returns_c, b_advantage_c
          

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      ratio = torch.exp(new_log_prob - b_log_prob) # added
      # ratio = b_log_prob/new_log_prob # added
      clipped_ratio = ratio.clamp(min=1.0 - eps,max=1.0 + eps) # added
      # pi_loss = torch.min(rt_theta*b_advantage,) # added
      pi_loss = -torch.min(ratio * b_advantage,clipped_ratio * b_advantage).mean() # added

      # Clipped value function objective
      clipped_value = b_value + (new_value - b_value).clamp(min=-eps, max=eps) # added
      # value_loss = (new_value - b_value)**2 # added
      value_loss = 0.5 * torch.max((new_value - b_returns) ** 2, (clipped_value - b_returns) **2).mean() # added

      # Entropy loss
      entropy_loss = -new_dist.entropy().mean() # added

      # Backpropagate losses
      loss = pi_loss + value_coef*value_loss + entropy_coef*entropy_loss # added
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps) # added

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  mean_rewards.append(storage.get_reward(normalized_reward=True))
  done_reward = sum((sum(storage.reward)/(sum(storage.done)+1)))/num_envs

  # TODO: If you never die implement an if statement that doesn't include the plus 1
  mean_rewards_done.append(done_reward)
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward(normalized_reward=True)}, \tMean reward done: {done_reward}')
  if first_loop:
    end = time()
    time_total = end-start
    estimated_time = (8e6/(8192/time_total))/3600
    print(f'Estimated time of completion: {estimated_time} hours')
  first_loop = False
  end = time()
  time_total = end-start
# Save the newest version after every epoch
  torch.save({
			'policy_state_dict': policy.state_dict(), # This is the policy
			'optimizer_state_dict': optimizer.state_dict(), # The optimizer used
			'Mean Reward': mean_rewards,
			'Mean Reward Done': mean_rewards_done,
			'Training time': time_total,
			}, f'/zhome/09/9/144141/Desktop/other_Deep_L/{model_name}.pt')
print(f'Completed training of {model_name}!')# Initialize the policy





