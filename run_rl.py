# Runs episodes, then executes 'train_q_models.py',
# which updates the model parameters, creates a new sub-directory for the
# action records, param_for_search, and search_results 

import sys
import subprocess
from rl_env import rl_env
import numpy as np
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
# from stable_baselines3.common.monitor import Monitor
from make_env import make_env
from stable_baselines3.common.vec_env import VecMonitor
from math import floor

# TODO: after init search pred still == 0 !?
# Now branch pred == 0 all the time, wtf, split the init file into two?

# move action rec and mode rec to run_1/ and repeat to see what happens to pred (i guess just branch pred for now)
# TODO: try dummy vec? at least when we move to p=1000, and might be worth a try on a smaller sample of p=5
# TODO: tensor Board


# TODO: 'terminal keys'?
# TODO: train over different lambdas
# TODO: change 'batch' to 'batch'
# TODO: write validation results to file, in each batch?
# TODO: compare pred and obs support
# TODO: action_record sub_dir by p (currently action_records/batch_{n}), model_rec


if __name__ ==  "__main__":
	first_batch_n = int(sys.argv[1]) # passed from command line when using and array job

	L0_list=[10**-2, 10**-3, 10**-4]

	for batch in range(first_batch_n, (first_batch_n + 8)):
		# Create batch sub-directories
		subprocess.run(f'mkdir action_records/batch_{batch}', shell=True)
		subprocess.run(f'mkdir model_records/batch_{batch}', shell=True)
		subprocess.run(f'mkdir param_for_search/batch_{batch}', shell=True)
		subprocess.run(f'mkdir results_of_search/batch_{batch}', shell=True)
		subprocess.run(f'mkdir model_copies/batch_{batch}', shell=True)
		subprocess.run(f'mkdir ep_res_records/batch_{batch}', shell=True)

		for L0 in L0_list:
			# Create L0 sub-directories
			log_L0 = -int(np.log10(L0))
			subprocess.run(f'mkdir param_for_search/batch_{batch}/L0_{log_L0}', shell=True)
			subprocess.run(f'mkdir results_of_search/batch_{batch}/L0_{log_L0}', shell=True)
			subprocess.run(f'mkdir model_copies/batch_{batch}/L0_{log_L0}', shell=True)

			# Make copies of q-models
			subprocess.run(f'cp -r models/* model_copies/batch_{batch}/L0_{log_L0}', shell=True)

	vec_env = SubprocVecEnv( \
	[make_env(batch_n=(first_batch_n + floor(i/3)), p=5, L0=L0_list[i%3]) for i in range(24)], \
	start_method='spawn')
	vec_env = VecMonitor(vec_env, filename="./monitor_logs")
	model = A2C('MlpPolicy', vec_env, verbose=1, tensorboard_log="./tb_logs/mini")
	model = model.save("sb_models/a2c_mini")

	# Load stable baselines rl agent
	model = A2C.load("sb_models/a2c_mini", vec_env)
	model.learn(total_timesteps=10000, log_interval=100, tb_log_name="run_1")
	model.save("sb_models/a2c_mini")

	# Update q-model parameters and sb agent
	for batch_n in range(first_batch_n + 8):
		subprocess.run(f'python train_q_models.py {batch_n}', shell=True)

