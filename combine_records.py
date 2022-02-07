import sys
import os
import numpy as np
import re

run_n = sys.argv[1]

for i in range(4):
	if i == 0: # branch
		in_dir = f'./action_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir) if re.match('branch*', f)]
		out_dir = f'./combined_action_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_rec_comb.npy'
	elif i == 1: # search
		# 'in_dir' and 'out_dir' as above
		record_list = [f for f in os.listdir(in_dir) if re.match('search*', f)]
		out_file_name = 'search_rec_comb.npy'
	
	elif i == 2: # model records
		in_dir = f'./model_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_model_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'model_rec_comb.npy'		

	elif i == 3: # episode result records
		in_dir = f'./ep_res_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_ep_res_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'ep_res_rec_comb.npy'

	array_list = []
	for file_name in record_list:
		try:
			record = np.load(f'{in_dir}/{file_name}')
			array_list.append(record)
		except ValueError as error:
			print(error)
			print(f'failed to open {file_name}')
	
	rec_comb = np.vstack(array_list)

	np.save(f'{out_dir}/{out_file_name}', rec_comb)
