import sys
import os
import re
import numpy as np
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import shutil

# A script to train the branch model and the search model on the combined
# action records found in './combined_action_records/run_<run_n>', where 
# <run_n> is passed to this script from the batch script (or command line) call 
# of 'python run_rl.py <run_n>'.


def update_param(run_n, n_layer, drop_out, regularization, learning_rate, \
		model_scope, reward_format, lean):
	# Load x and y
	if lean:
		record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb_lean.npy')
	else:
		record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb.npy')

	if model_scope == 'specific':
		# Filter for specific conditions: L0=1e-3, L2=1e-3, p=100
		# L0, L2, and p are the first 3 columns of the records
		row_indexes = []
		for i in range(record.shape[0]):
			if record[i,0] == 1e-3 and record[i,1] == 1e-3 and record[i,2]==100:
				row_indexes.append(i)
		
		record = record[row_indexes,:]

	n_col = record.shape[1]
	x, y = np.hsplit(record, np.array([n_col-1]))
	y = y.reshape(-1)

	# Make vector of weights (to up-weight positive y values)
	n_obs = x.shape[0]
	if reward_format == 'big_binary':
		n_pos = sum(y>0.01)
	else:
		n_pos = sum(y>10**-6)
	if n_pos > 0:
		weight_pos = n_obs / n_pos
	else:
		weight_pos = 1

	weights = np.ones(n_obs)

	if reward_format == 'big_binary':
		weights[y>0.01] = np.full(shape=n_pos, fill_value=weight_pos)
	else:
		weights[y>10**-6] = np.full(shape=n_pos, fill_value=weight_pos)

	# Make binary reward
	if reward_format == 'binary':
		y[y>10**-6] = np.ones(n_pos)

	if reward_format == 'big_binary':
		y[y>0.01] = np.ones(n_pos)

	if reward_format == 'numeric':
		y = y*100

	# Load model
	if lean:
		model_name = f'lean_in{x.shape[1]}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_{model_scope}'
	else:
		model_name = f'branch_model_in{x.shape[1]}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_{model_scope}'

	model = tf.keras.models.load_model(f'./models/{model_name}')

	# Make or clear log directory
	if lean:
		log_dir = f"tb_logs/lean/run_{run_n}/{model_name}"
	else:
		log_dir = f"tb_logs/run_{run_n}/{model_name}"
	os.makedirs(log_dir, exist_ok=True)
	for sub_dir in os.listdir(log_dir): 
		# the automatically created sub_dirs are 'train' and 'validation'
		shutil.rmtree(os.path.join(log_dir, sub_dir))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		min_delta=0.001,
		patience=50,
		mode='min',
		restore_best_weights=True)

	# Train model
	model.fit(x, y, epochs=500, verbose=0, sample_weight=weights, \
		callbacks=[tensorboard_callback, early_stopping_callback], \
		validation_split=0.1, \
		# validation_data = [validation_x, validation_y, validation_weights], \
		# validation_freq=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 100, 200, 300])
		validation_freq=1)

	model.save(f'./models/{model_name}')

if __name__ == '__main__':
	update_param(	run_n=sys.argv[1], \
			n_layer = sys.argv[2], \
			drop_out = sys.argv[3], \
			regularization = sys.argv[4], \
			learning_rate = sys.argv[5], \
			model_scope = sys.argv[6], \
			reward_format = sys.argv[7], \
			lean = sys.argv[8])
