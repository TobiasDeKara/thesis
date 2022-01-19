# Runs 100 episodes, then executes 'train_q_models.py',
# which updates the model parameters, creates a new sub-directory for the
# action records, param_for_search, and search_results 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import subprocess
from rl_env import rl_env
import numpy as np
from model_performance import get_mse
import gym
from stable_baselines3 import A2C

# TODO: train over different lambdas
# TODO: change 'epoch' to 'batch'
# TODO: get other model stats: n_non_zero, mean non_zero (pred and obs)
# TODO: make predictions and get mse on validation data (epoch/batch 1)
# TODO: write validation results to file
# TODO: compare pred and obs support
# TODO: action_record sub_dir (currently action_records/epoch_{n}), model_rec



# Create n_th epoch sub-directories for action_records, model_records, 
# param_for_search, and search_results
epoch_n = int(sys.argv[1])
# print(f'epoch_n: {epoch_n}')

subprocess.run(f'cd action_records; mkdir epoch_{epoch_n}', shell=True)
subprocess.run(f'cd model_records; mkdir epoch_{epoch_n}', shell=True)
subprocess.run(f'cd param_for_search; mkdir epoch_{epoch_n}', shell=True)
subprocess.run(f'cd results_of_search; mkdir epoch_{epoch_n}', shell=True)

# Initialize rl_env
env = rl_env(p=5, l0=10**-3, epoch = epoch_n)

# model = A2C("MlpPolicy", env, verbose=1)
model = A2C.load("sb_models/a2c_mini")

# n episodes
for i in range(100):
	obs = env.reset()
	done = False
	while(done==False):
		action, _states = model.predict(obs)
		obs, reward, done, info = env.step(action) 

# Update q-model parameters
subprocess.run(f'python train_q_models.py {epoch_n}', shell=True)

# Save stable_baselines agent model		
model.save("sb_models/a2c_mini")

# print(get_mse(epoch_n))
