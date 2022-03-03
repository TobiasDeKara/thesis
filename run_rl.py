# Runs episodes,
# creates a new sub-directories for the
# action records, param_for_search, and search_results 

import os
import sys
import subprocess
from rl_env import rl_env
import numpy as np

if __name__ ==  "__main__":
	run_n = sys.argv[1] # passed from command line when using an array job
	batch_n = int(int(sys.argv[2])/3)
	log_L0 = (int(sys.argv[2]) % 3) + 2
	
	# Create run sub-directories for records
	os.makedirs(f'action_records/run_{run_n}', exist_ok=True)
	os.makedirs(f'model_records/run_{run_n}', exist_ok=True)
	os.makedirs(f'ep_res_records/run_{run_n}', exist_ok=True)
	
	# Create L0 sub-directories
	# log_L0 = -int(np.log10(L0))
	os.makedirs(f'param_for_search/batch_{batch_n}/L0_{log_L0}', exist_ok=True)
	os.makedirs(f'results_of_search/batch_{batch_n}/L0_{log_L0}', exist_ok=True)
	os.makedirs(f'model_copies/batch_{batch_n}/L0_{log_L0}', exist_ok=True)

	# Make copies of q-models
	subprocess.run(f'cp -r models/* model_copies/batch_{batch_n}/L0_{log_L0}', shell=True)

	# Run RL env
	env = rl_env(p=1000, L0=10**-log_L0, batch_n=batch_n, run_n=run_n)
	env.reset()
	done = False

	for step_n in range(3000):
		if done == False:
			done, info = env.step()
		elif done == True:
			env.reset()
			done, info = env.step()
