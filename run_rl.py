# Runs episodes, then executes 'train_q_models.py',
# which updates the stable baselines rl agent model parameters, 
# creates a new sub-directory for the
# action records, param_for_search, and search_results 

import sys
import subprocess
from rl_env import rl_env
import numpy as np
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from make_env import make_env
from stable_baselines3.common.vec_env import VecMonitor

if __name__ ==  "__main__":
	run_n = int(sys.argv[1]) # passed from command line when using and array job
	first_batch_n = int(sys.argv[2])

	L0_list=[10**-2, 10**-3, 10**-4]
	
	# Create run sub-directories for records
	subprocess.run(f'mkdir action_records/run_{run_n}', shell=True)
	subprocess.run(f'mkdir model_records/run_{run_n}', shell=True)
	subprocess.run(f'mkdir ep_res_records/run_{run_n}', shell=True)

	for batch in range(first_batch_n, (first_batch_n + 8)):
		# Create batch sub-directories
		subprocess.run(f'mkdir param_for_search/batch_{batch}', shell=True)
		subprocess.run(f'mkdir results_of_search/batch_{batch}', shell=True)
		subprocess.run(f'mkdir model_copies/batch_{batch}', shell=True)

		for L0 in L0_list:
			# Create L0 sub-directories
			log_L0 = -int(np.log10(L0))
			subprocess.run(f'mkdir param_for_search/batch_{batch}/L0_{log_L0}', shell=True)
			subprocess.run(f'mkdir results_of_search/batch_{batch}/L0_{log_L0}', shell=True)
			subprocess.run(f'mkdir model_copies/batch_{batch}/L0_{log_L0}', shell=True)

			# Make copies of q-models
			subprocess.run(f'cp -r models/* model_copies/batch_{batch}/L0_{log_L0}', shell=True)

	vec_env = DummyVecEnv([make_env(batch_n=(first_batch_n + int(i/3)), run_n=run_n, p=5, L0=L0_list[i%3]) for i in range(24)])
	vec_env = VecMonitor(vec_env, filename="./monitor_logs")

	# Load stable baselines rl agent
	model = A2C.load("sb_models/a2c_mini", vec_env, tensorboard_log='./tb_logs/sb')
	model.learn(total_timesteps=50000, log_interval=100, tb_log_name=f'run_{run_n}', reset_num_timesteps=False)
	model.save("sb_models/a2c_mini")
