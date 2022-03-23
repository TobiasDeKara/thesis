# Runs episodes,
# creates a new sub-directories for the
# action records, param_for_search, and search_results 

import os
import sys
import subprocess
from rl_env import rl_env
import numpy as np

if __name__ ==  "__main__":
	p = int(sys.argv[1])
	log_p = int(np.log10(p))
	run_n = sys.argv[2]
	batch_n = sys.argv[3]
	log_L0 = int(int(sys.argv[4])/3) + 2
	log_L2 = (int(sys.argv[4]) % 3) + 2
	
	# Create run sub-directories for records
	os.makedirs(f'action_records/run_{run_n}', exist_ok=True)
	os.makedirs(f'model_records/run_{run_n}', exist_ok=True)
	os.makedirs(f'ep_res_records/run_{run_n}', exist_ok=True)
	
	# Create L0 sub-directories
	# log_L0 = -int(np.log10(L0))
	os.makedirs(f'param_for_search/p_{log_p}/batch_{batch_n}/L0_{log_L0}_L2_{log_L2}', exist_ok=True)
	os.makedirs(f'results_of_search/p_{log_p}/batch_{batch_n}/L0_{log_L0}_L2_{log_L2}', exist_ok=True)
	os.makedirs(f'model_copies/batch_{batch_n}/L0_{log_L0}_L2_{log_L2}', exist_ok=True)

	# Run RL env
	model_name = 'branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range'
	# model_name = 'branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_specific'
	env = rl_env(p=p, L0=10**-log_L0, L2=10**-log_L2, batch_n=batch_n, run_n=run_n, \
		branch_model_name=model_name, greedy_epsilon=0)
	max_n_step = p # 100 # used during training

	for ep_n in range(90):  # 100 train (or 10 test) ep x 3 corr. (aka rho) x 3 SNR
		n_step = 0
		env.reset()
		done = False
		while not done and n_step < max_n_step:
			done, info = env.step()
			n_step += 1
