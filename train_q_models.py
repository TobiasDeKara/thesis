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


def update_param(reward_format, n_layer, drop_out, regularization, learning_rate, \
		run_n=sys.argv[1], specific=False):
	# Load x and y
	if specific:
		record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb_specific.npy')
	elif run_n == 'all':
		record = np.load(f'./combined_action_records/all_runs/all_branch_records.npy')
	else:
		record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb.npy')

	# validation_record = np.load(f'./combined_action_records/run_validation/branch_rec_comb.npy')
	n_col = record.shape[1]
	x, y = np.hsplit(record, np.array([n_col-1]))
	# validation_x, validation_y = np.hsplit(validation_record, np.array([n_col-1]))
	y = y.reshape(-1)
	# validation_y = validation_y.reshape(-1)

	# Make vector of weights (to up-weight positive y values)
	n_obs = x.shape[0]
	# validation_n_obs = validation_x.shape[0]
	n_pos = sum(y>10**-6)
	# validation_n_pos = sum(validation_y > 10**-6)
	if n_pos > 0:
		weight_pos = n_obs / n_pos
	else:
		weight_pos = 1
	# if validation_n_pos > 0:
	#	validation_weight_pos = validation_n_obs / validation_n_pos
	# else:
	#	validation_weight_pos = 1

	weights = np.ones(n_obs)
	# validation_weights = np.ones(validation_n_obs)

	weights[y>10**-6] = np.full(shape=n_pos, fill_value=weight_pos)
	# validation_weights[validation_y>10**-6] = \
	# np.full(shape=validation_n_pos, fill_value=validation_weight_pos)

	# Make binary reward
	if reward_format == 'binary':
		y[y>10**-6] = np.ones(n_pos)
		# validation_y[validation_y>10**-6] = np.ones(validation_n_pos)

	# Load model
	if specific:
		model_name = f'branch_model_in{x.shape[1]}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_specific'
	else:
		model_name = f'branch_model_in{x.shape[1]}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_specific'

	model = tf.keras.models.load_model(f'./models/{model_name}')

	# Make or clear log directory
	if run_n == 'all':
		log_dir = f"tb_logs/all_runs/{model_name}"
	else:
		log_dir = f"tb_logs/run_{run_n}/{model_name}"
	os.makedirs(log_dir, exist_ok=True)
	for sub_dir in os.listdir(log_dir): 
		# the automatically created sub_dirs are 'train' and 'validation'
		shutil.rmtree(os.path.join(log_dir, sub_dir))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

	# Train model
	model.fit(x, y, epochs=10000, verbose=0, sample_weight=weights, callbacks=[tensorboard_callback], \
		validation_split=0.1, \
		# validation_data = [validation_x, validation_y, validation_weights], \
		# validation_freq=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 100, 200, 300]
		validation_freq=100)

	model.save(f'./models/{model_name}')

if __name__ == '__main__':
	n_layer = sys.argv[2]
	drop_out = sys.argv[3]
	regularization = sys.argv[4]
	learning_rate = sys.argv[5]
	specific = sys.argv[6]
	update_param(reward_format = 'binary', \
		n_layer=n_layer, drop_out=drop_out, regularization=regularization, \
		learning_rate=learning_rate, specific=specific)
#	update_param(reward_format = 'numeric', \
#		n_layer=n_layer, drop_out=drop_out)
