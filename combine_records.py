import os
import numpy as np
import re

run_n = 0
for i in range(2):
	if i == 0: # branch
		record_list = [f for f in os.listdir(f'./action_records/run_{run_n}') if re.match('branch*', f)]
		file_name = './combined_action_records/branch_rec_comb.npy'
	else: # search
		record_list = [f for f in os.listdir(f'./action_records/run_{run_n}') if re.match('search*', f)]
		file_name = './combined_action_records/search_rec_comb.npy'

	array_list = []
	for file_name in record_list:
		try:
			record = np.load(f'./action_records/run_{run_n}/{file_name}')
			array_list.append(record)
	except ValueError as error:
		print(error)
		print(f'failed to open {file_name}')
	
	rec_comb = np.vstack(array_list)

	np.save(file_name, rec_comb)
