import numpy as np
import pandas as pd
import subprocess
import os
import tensorflow as tf

# For reference
# model_record = np.array([run_n, action.step_number,  \
#	action.q_hat[0], action.frac_change_in_opt_gap], dtype = float)

def get_run_stats(run_n=0, model_name=None):

	if model_name is not None:
		# Get y and predictions using given model
		if run_n == 'all':
			action_rec = \
				np.load('./combined_action_records/all_runs/branch_rec_comb.npy')
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

		n_col = model_rec.shape[1]
		pred = model_rec[:, n_col-2]
		y = model_rec[:, n_col-1]

	# Calculate stats
	mse = ((y-pred)**2).mean()
	n_obs = y.shape[0]
	
	n_non_zero_obs = (y > 0).sum()
	n_non_zero_pred = (pred > 0).sum()
   
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
