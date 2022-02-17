import gym
from rl_env import rl_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv # SubprocVecEnv, 
from typing import Callable
from stable_baselines3.common.vec_env import VecMonitor
from make_env import make_env
import sys

# For stable baselines agent
if __name__ ==  "__main__":
	L0_list = [10**-2, 10**-3, 10**-4]
	p = 1000
	run_n = sys.argv[1]

	vec_env = \
	DummyVecEnv([make_env(run_n=run_n, batch_n=int(i/3), p=p, L0=L0_list[i%3]) for i in range(24)])

	vec_env = VecMonitor(vec_env, filename="./monitor_logs")

	model = A2C('MlpPolicy', vec_env, verbose=1, tensorboard_log="./tb_logs/sb")
	model = model.save("sb_models/a2c_p3")
