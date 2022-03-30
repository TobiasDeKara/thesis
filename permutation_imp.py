from copy import copy
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys


rep_n = sys.argv[1]

run_n = 0
record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb.npy')
n_col = record.shape[1]
x, y = np.hsplit(record, np.array([n_col-1]))
y = y.reshape(-1)

n_obs = x.shape[0]
n_col = x.shape[1]
n_pos = sum(y>10**-6)
if n_pos > 0:
	weight_pos = n_obs / n_pos
else:
	weight_pos = 1

weights = np.ones(n_obs)
weights[y>10**-6] = np.full(shape=n_pos, fill_value=weight_pos)

y[y>10**-6] = np.ones(n_pos)

model_name = 'branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range'

model = tf.keras.models.load_model(f'./models/{model_name}')

res_all_vars = model.evaluate(x=x, y=y, sample_weight=weights)

res_list = []
for i in range(n_col):
	new_x = x.copy()
	perm = np.random.choice(n_obs, size=n_obs, replace=False)
	new_x[:,i] = new_x[perm,i]
	res = model.evaluate(x=new_x, y=y, sample_weight=weights, verbose=0)
	res_list.append(res)

res_list.append(res_all_vars)
res_list = np.array(res_list)
np.savetxt(f'permutation_mse_list_rep_{rep_n}.csv', res_list, delimiter=',')


