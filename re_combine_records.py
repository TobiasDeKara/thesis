import os
import re
import numpy as np

for record_type in ['action', 'ep_res', 'model']:
	main_dir = f'./combined_{record_type}_records'
	run_dirs_list = [d for d in os.listdir(main_dir) if re.match('run_[0-9]', d)]
	out_dir = f'./combined_{record_type}_records/all_runs'
	os.makedirs(out_dir, exist_ok=True)

	array_list = []
	if record_type == 'action':
		file_name = 'branch_rec_comb.npy'
		out_file_name = 'all_branch_records.npy'
	elif record_type == 'ep_res':
		file_name = 'ep_res_rec_comb.npy'
		out_file_name = 'all_ep_res_rec_comb.npy'
	else:
		file_name = 'branch_model_rec_comb.npy'
		out_file_name = 'all_branch_model_rec_comb.npy'

	for run_dir in run_dirs_list:
		try:
			record = np.load(f'{main_dir}/{run_dir}/{file_name}')
			array_list.append(record)
		except FileNotFoundError as error:
			print(error)
			print(f'failed to open {file_name}')

	if array_list:
		rec_comb = np.vstack(array_list)
		np.save(f'{out_dir}/{out_file_name}', rec_comb)
