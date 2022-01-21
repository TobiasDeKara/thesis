# Runs 100 episodes, then executes 'train_q_models.py',
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


# TODO: the code that is working below is not yet working in init..., error reset by peer or something
# We want to change the batch_run_rl.sh to 24 nodes, and not an array (for now), for now pass 1 as the arg

# TODO: train over different lambdas
# TODO: change 'epoch' to 'batch'
# TODO: write validation results to file, in each batch?
# TODO: compare pred and obs support
# TODO: action_record sub_dir by p (currently action_records/epoch_{n}), model_rec

# Citation: The following function is a based on code from 
# 'Stable Baselines3 - Easy Multiprocessing' available at 
# https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb#scrollTo=pUWGZp3i9wyf
# Accessed 1/21/22
def make_env(rank: int, first_epoch_n: int=1) -> Callable:
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    """
    def _init() -> gym.Env:
        epoch_n = first_epoch_n + rank
        env = rl_env(p=5, l0=10**-3, epoch=epoch_n) 
        return env
    return _init

if __name__ ==  "__main__":
	num_cpu = 24
	first_epoch_n = int(sys.argv[1]) # passed from command line when using and array job
	vec_env = SubprocVecEnv( \
		[make_env(rank=i, first_epoch_n=first_epoch_n) for i in range(num_cpu)], \
		start_method='spawn')
	# model = A2C('MlpPolicy', vec_env, verbose=0)

	# Load stable baselines rl agent
	model = A2C.load("sb_models/a2c_mini", env)
	model.learn(total_timestpes=3000, log_interval=1000)
	model.save("sb_models/a2c_mini")

	# Update q-model parameters and sb agent
	for epoch_n in range(first_epoch_n + num_cpu):
		subprocess.run(f'python train_q_models.py {epoch_n}', shell=True)

