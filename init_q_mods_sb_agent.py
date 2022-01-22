import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from rl_env import rl_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable

# For Branhcing
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(60,)) ) # 46 static stats plus 14 branch-specific stats
model.add(layers.Dense(1, activation="relu"))

optimizer = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss= "mean_squared_error")
print(model.summary())
model.save('./models/branch_model_in60_lay2')

# For Searching
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(51,)) ) # 46 static stats plus 5 search-node-specific
model.add(layers.Dense(1, activation="relu"))
optimizer = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss= "mean_squared_error")
print(model.summary())
model.save('./models/search_model_in51_lay2')

# For stable baselines agent
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
	first_epoch_n = 1
	vec_env = SubprocVecEnv( \
		[make_env(rank=i, first_epoch_n=first_epoch_n) for i in range(num_cpu)], \
		start_method='spawn')
	model = A2C('MlpPolicy', vec_env, verbose=1)
	model = model.save("sb_models/a2c_mini")
