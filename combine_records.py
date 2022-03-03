import sys
import os
import numpy as np
import re

run_n = sys.argv[1]

for i in [0, 1, 2]:
	if i == 0: # branch actions 
		in_dir = f'./action_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir) if re.match('branch', f)]
		out_dir = f'./combined_action_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_rec_comb.npy'
	
	elif i == 1: # branch model records
		in_dir = f'./model_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir) if re.match('branch', f)]
		out_dir = f'./combined_model_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_model_rec_comb.npy'		

	elif i == 2: # episode result records
		in_dir = f'./ep_res_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_ep_res_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'ep_res_rec_comb.npy'

	elif i == 3: # seed_support records
		in_dir = f'./synthetic_data/p3/seed_support_records'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_seed_support_records/'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'seed_support_rec_comb.npy'

	array_list = []
	for file_name in record_list:
		try:
			record = np.load(f'{in_dir}/{file_name}')
			if not (i==2 and record.shape[0]==5): # for compatibility
				array_list.append(record)
		except ValueError as error:
			print(error)
			print(f'failed to open {file_name}')

	if array_list:
		rec_comb = np.vstack(array_list)

		np.save(f'{out_dir}/{out_file_name}', rec_comb)
