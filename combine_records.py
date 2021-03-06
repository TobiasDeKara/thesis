import sys
import os
import numpy as np
import re

run_n = sys.argv[1]

for i in [0,1,2]:
	if i == 0: # branch actions 
		in_dir = f'./action_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]

		out_dir = f'./combined_action_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = 'branch_rec_comb.npy'
	
	elif i == 1: # branch model records
		# Example action record file name:
		# branch_model_rec_dim7_gen_syn_n3_p1_corr0.3_snr10.0_seed130396699L0_4_L2_3_branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range_1.npy

		in_dir = f'./model_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_model_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = f'branch_model_rec_comb.npy'		

	elif i == 2: # episode result records
		# example file name:
		# ep_res_rec_gen_syn_n3_p1_corr0.3_snr10.0_seed218716013L0_2_L2_2_branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range.npy

		in_dir = f'./ep_res_records/run_{run_n}'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_ep_res_records/run_{run_n}'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = f'ep_res_rec_comb.npy'

	elif i == 3: # seed_support records
		p = 4  ### Set each time ###
		in_dir = f'./synthetic_data/p{p}/seed_support_records'
		record_list = [f for f in os.listdir(in_dir)]
		out_dir = f'./combined_seed_support_records/'
		os.makedirs(out_dir, exist_ok=True)
		out_file_name = f'seed_support_rec_comb_p{p}.npy'

	array_list = []
	info_list = []
	for file_name in record_list:
		try:
			record = np.load(f'{in_dir}/{file_name}')
			array_list.append(record)

			# include info from file name
			if i == 1 or i == 2: 
				corr = re.search('corr(...)', file_name).group(1)
				snr = re.search('snr(\d*)', file_name).group(1)
				model_type = re.search('(range|specific)', file_name).group(1)
				reward_fmt = re.search('(binary|numeric|sig_num)', file_name).group(1)

				info = np.array([corr, snr, model_type, reward_fmt])

				if i == 1:  # model records have multiple obs. per rec.
					n_rows_rec = record.shape[0]
					info = np.tile(info, (n_rows_rec,1))

				info_list.append(info)	

		except ValueError as error:
			print(error)
			print(f'failed to open {file_name}')

	if array_list:
		rec_comb = np.vstack(array_list)

		if i == 1 or i == 2:
			info_comb = np.vstack(info_list)
			rec_comb = np.hstack([rec_comb, info_comb])

		np.save(f'{out_dir}/{out_file_name}', rec_comb)
