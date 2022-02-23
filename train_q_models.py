import sys
import os
import re
import numpy as np
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# A script to train the branch model and the search model on the combined
# action records found in './combined_action_records/run_<run_n>', where 
# <run_n> is passed to this script from the batch script (or command line) call 
# of 'python run_rl.py <run_n>'.


def update_param(action_type, reward_format, n_layer, drop_out, run_n=sys.argv[1]):
	# Load x and y
	record = np.load(f'./combined_action_records/run_{run_n}/{action_type}_rec_comb.npy')
	validation_record = np.load(f'./combined_action_records/run_validation/{action_type}_rec_comb.npy')
	n_col = record.shape[1]
	x, y = np.hsplit(record, np.array([n_col-1]))
	validation_x, validation_y = np.hsplit(validation_record, np.array([n_col-1]))
	y = y.reshape(-1)
	validation_y = validation_y.reshape(-1)

	# Make vector of weights (to up-weight positive y values)
	n_obs = x.shape[0]
	validation_n_obs = validation_x.shape[0]
	n_pos = sum(y>0)
	validation_n_pos = sum(validation_y > 0)
	if n_pos > 0:
		weight_pos = n_obs / n_pos
	else:
		weight_pos = 1
	if validation_n_pos > 0:
		validation_weight_pos = validation_n_obs / validation_n_pos
	else:
		validation_weight_pos = 1

	weights = np.ones(n_obs)
	validation_weights = np.ones(validation_n_obs)

	weights[y>0] = np.full(shape=n_pos, fill_value=weight_pos)
	validation_weights[y>0] = np.full(shape=validation_n_pos, fill_value=validation_weight_pos)

	# Make binary reward
	if reward_format == 'binary':
		y[y>0] = np.ones(n_pos)
		validation_y[validation_y>0] = np.ones(validation_n_pos)

	# Load model
	model_name = f'{action_type}_model_in{x.shape[1]}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}'
	model = tf.keras.models.load_model(f'./models/{model_name}')
	log_dir = f"tb_logs/q_{action_type}/run_{run_n}/{reward_format}/lay{n_layer}_drop_out_{drop_out}"
	os.makedirs(log_dir, exist_ok=True)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

	model.fit(x, y, epochs=10, verbose=0, sample_weight=weights, callbacks=[tensorboard_callback], \
	validation_data = [validation_x, validation_y, validation_weights)
	model.save(f'./models/{model_name}')

if __name__ == '__main__':
	for n_layer in [3, 4]:
		for drop_out in ['yes', 'no']:
			update_param(action_type='branch', reward_format = 'binary', \
			n_layer=n_layer, drop_out=drop_out)
			update_param(action_type='branch', reward_format = 'numeric', \
			n_layer=n_layer, drop_out=drop_out)
			update_param(action_type='search', reward_format = 'binary', \
			n_layer=n_layer, drop_out=drop_out)
			update_param(action_type='search', reward_format = 'numeric', \
			n_layer=n_layer, drop_out=drop_out)
