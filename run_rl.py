# Runs 1000 episodes, then executes 'train_q_models.py',
# which updates the model parameters, creates a new sub-directory for the
# action records, 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
from rl_env import rl_env
import numpy as np
from model_performance import get_mse
import gym
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import ACKTR


# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import ACKTR
# 
# # multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)
# 
# model = ACKTR(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("acktr_cartpole")
# 
# del model # remove to demonstrate saving and loading
# 
# model = ACKTR.load("acktr_cartpole")
# 
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)



# Create n_th epoch sub-directories for action_records and model_records
n_prev_epoch = subprocess.run('cd action_records; ls -d1U */ | wc -l', capture_output=True, \
	text=True, shell=True).stdout

epoch_n = int(n_prev_epoch) + 1

subprocess.run(f'cd action_records; mkdir epoch_{epoch_n}', shell=True)
subprocess.run(f'cd model_records; mkdir epoch_{epoch_n}', shell=True)

# Initialize rl_env
env = rl_env(p=5, l0=10**-3, epoch = epoch_n)

# n episodes
for i in range(1000):
	obs = env.reset()
	done = False
	while(done==False):
		# obs, reward, done, info = env.step(np.random.choice(2)) # Add agent 
		obs, reward, done, info = env.step(0)
		
exec(open('train_q_models.py').read())

get_mse(3)
