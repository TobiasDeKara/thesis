import re
import numpy as np
import pandas as pd
import subprocess
import os
import tensorflow as tf

# For reference
# model_record = np.array([run_n, p, L0, L2, action.step_number,  \
#        action.q_hat[0], action.frac_change_in_opt_gap, corr, snr, model_type, reward_format])

# Example file name:
# branch_model_rec_dim7_gen_syn_n3_p1_corr0.3_snr10.0_seed130396699L0_2_L2_3_branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range_0.npy


def get_run_stats(run_n=0, model_name=None):

	if model_name is not None:
		# Get y and predictions using given model
		if run_n == 'all':
			action_rec = \
				np.load('./combined_action_records/all_runs/all_branch_records.npy')
		else:
			action_rec = \
				np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb.npy')

		n_obs =  action_rec.shape[0]
		n_col = action_rec.shape[1]

		x, y = np.hsplit(action_rec, np.array([n_col-1]))

		model = tf.keras.models.load_model(f'./models/{model_name}')
		pred = model.predict(x, verbose=0)

	else:
		# Get y and predictions from model used during training
		rec_file_name = f'./combined_model_records/run_{run_n}/branch_model_rec_comb.npy'
		model_rec = np.load(rec_file_name)

		pred = model_rec[:, 5].astype(float)
		y = model_rec[:, 6].astype(float)

	# Calculate stats
	n_obs = y.shape[0]
	
	n_non_zero_obs = (y > 10**-6).sum()
	n_non_zero_pred = (pred > 10**-6).sum()

	if re.search(model_name, 'numeric'):
		y = y*100
	if re.search(model_name, 'binary'):
		y[y > 10**-6] = np.ones(n_non_zero_obs)

	mse = ((y-pred)**2).mean()
   
	# Note for next line, sum of non-zero elements == sum of all elements
	mean_non_zero_obs = y.sum() / n_non_zero_obs 
	mean_non_zero_pred = pred.sum() / n_non_zero_pred

	# Gather return values
	out = pd.DataFrame([[n_obs, mse], 
	[y.min(), pred.min()],
	[y.mean(), pred.mean()],
	[y.max(), pred.max()],
	[mean_non_zero_obs, mean_non_zero_pred], 
	[n_non_zero_obs, n_non_zero_pred]],
	index = ['n_obs_and_mse', 'min', 'mean', 'max', 'mean_non_zero_value', 'n_non_zero_values'], 
	columns=['obs', 'pred'])

	return(out)
