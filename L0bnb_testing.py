import numpy as np
from l0bnb import BNBTree
import sys
import os
import re

batch_n = 1 # testing data

p = int(sys.argv[1])
log_p = int(np.log10(p))
p_sub_dir = f'p{log_p}'

branching='maxfrac'
#branching='strong'

model_type = 'L0bnb_' + branching

data_dir = f'synthetic_data/{p_sub_dir}/batch_{batch_n}'

x_file_list = [f for f in os.listdir(data_dir) if re.match('x', f)]

res_list = []

for i in range(len(x_file_list)):
	x_file_name = x_file_list[i]
	corr = re.search('corr(...)', x_file_name).group(1)
	snr = re.search('snr(\d*)', x_file_name).group(1)

	y_file_name = x_file_name.replace('x', 'y')
	x = np.load(f'synthetic_data/{p_sub_dir}/batch_{batch_n}/{x_file_name}')
	y = np.load(f'synthetic_data/{p_sub_dir}/batch_{batch_n}/{y_file_name}')
	y = y.reshape(1000)

	for L0 in [0.01, 0.001, 0.0001]:
		for L2 in [0.01, 0.001, 0.0001]:
			# print(L0, L2)
			tree = BNBTree(x, y)
		
			# tree_sol = tree.solve(
			# current_lambda_0,
			# lambda_2,
			# m,
			# gap_tol=gap_tol,
			# time_limit=time_limit)
		
			tree_sol = tree.solve(
				L0,
				L2,
				5,
				gap_tol=0.01,
				time_limit=7200,
				branching=branching
				)
		
			len_model_support = sum(tree_sol[1] > 0)
			ep_res_record = np.array([p, L0, L2, corr, snr, \
						tree.number_of_nodes-1, \
						len_model_support])
		
			res_list.append(ep_res_record)
			# print(tree.number_of_nodes-1)
np.save(f'./L0bnb_res_records/L0bnb_testing_p{log_p}_branch_{branching}', np.vstack(res_list))

