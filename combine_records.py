import sys
import os
import numpy as np
import re

run_n = sys.argv[1]

# TODO: change ep_res_recs to include support size, and % of true support included by loading b info,
# and change b info to a making a numpy array in gen_syn_data

for i in range(2): # TODO: change range to 3 after changing ep_res_records
	if i == 0: # branch
		record_list = [f for f in os.listdir(f'./action_records/run_{run_n}') if re.match('branch*', f)]
		in_dir = f'./action_records/run_{run_n}'
		out_dir = f'./combined_action_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_rec_comb.npy'
	elif i == 1: # search
		record_list = [f for f in os.listdir(f'./action_records/run_{run_n}') if re.match('search*', f)]
		# 'in_dir' and 'out_dir' as above
		out_file_name = 'search_rec_comb.npy'

	else: # episode result records
		record_list = [f for f in os.listdir(f'./ep_res_records/run_{run_n}')]
		in_dir = f'./ep_res_records/run_{run_n}'
		out_dir = f'./combined_ep_res_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_rec_comb.npy'

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
