# Runs episodes,
# creates a new sub-directories for the
# action records, param_for_search, and search_results 

import os
import sys
import subprocess
from rl_env import rl_env
import numpy as np
from jug import TaskGenerator
import re
import time

start_time = time.time()
p = 50
log_p = 1
run_n = 80
batch_n = 2
log_L0 = 2
log_L2 = 2
max_n_step = 22
test_train = 'train'
model_scope = 'range'
reward_format = 'binary'

# p = int(sys.argv[1])
# log_p = int(np.log10(p))
# run_n = sys.argv[2]
# batch_n = sys.argv[3]
# log_L0 = int(int(sys.argv[4])/3) + 2
# log_L2 = (int(sys.argv[4]) % 3) + 2
# max_n_step= int(sys.argv[5])
# test_train = sys.argv[6] # {'test', 'train'} used to set number of episodes and greedy_epsilon here, and passed to rl_env to not record action records of testing episodes.
# model_scope = sys.argv[7]
# assert model_scope in ['range', 'specific'], \
# 	f'model_scope expected range or specific, got {model_scope}'
# reward_format = sys.argv[8]
# assert reward_format in ['binary', 'numeric', 'big_binary', 'sig_num'], \
# 	f'reward_format expeceted binary, numeric, big_binary, sig_num, got {reward_format}'


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
model_name = f'branch_model_in62_lay6_drop_out_yes_rew_{reward_format}_reg_True_rate_1e-05_{model_scope}'

data_dir = f'/gpfs/scratch/tdekara/synthetic_data/p{log_p}/batch_{batch_n}'
x_file_list = [f for f in os.listdir(data_dir) if re.match('x', f)]

if test_train == 'test':
	greedy_epsilon=0
else:
	greedy_epsilon=0.3

@TaskGenerator
def run_1_ep(p,log_L0,log_L2,batch_n,run_n,model_name,greedy_epsilon,test_train,x_file_name,max_n_step):
	env = rl_env(p=p, L0=10**-log_L0, L2=10**-log_L2, batch_n=batch_n, run_n=run_n, \
		branch_model_name=model_name, greedy_epsilon=greedy_epsilon, test_train=test_train)
	env.reset(x_file_name=x_file_name)
	n_step = 0
	done = False
	while not done and n_step < max_n_step:
		done = env.step()
		n_step += 1

for x_file_name in x_file_list[0:10]:  # change after testing
	run_1_ep(p,log_L0,log_L2,batch_n,run_n,model_name,greedy_epsilon,test_train,x_file_name,max_n_step)


# print(start_time-time.time())
