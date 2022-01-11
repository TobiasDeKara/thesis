# One episode

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from rl_env import rl_env
import numpy as np

env = rl_env(p=5, l0=10**-3)

obs = env.reset()
done = False
while(done==False):
	obs, reward, done, info = env.step(np.random.choice(2))
