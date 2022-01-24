# Citation: The following function is a based on code from 
# 'Stable Baselines3 - Easy Multiprocessing' available at 
# https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb#scrollTo=pUWGZp3i9wyf
# Accessed 1/21/22

import gym
from typing import Callable
from rl_env import rl_env

def make_env(batch_n, p=5, L0=10**-3) -> Callable:
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess (starting at zero)
    """
    def _init() -> gym.Env:
        env = rl_env(p=p, L0=L0, batch_n=batch_n)
        return env
    return _init
