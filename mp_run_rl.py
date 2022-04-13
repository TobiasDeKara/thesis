import sys
from itertools import repeat
import re
from multiprocessing import Pool
from multiprocessing import get_context
import os
import numpy as np
from rl_env import rl_env
import tensorflow as tf


def run_1_ep(x_file_name, L0, L2, p, batch_n,  model_name):
	env = rl_env(p=p, L0=L0, L2=L2, batch_n=batch_n, run_n='replay_buffer', \
		branch_model_name=model_name, greedy_epsilon=0.3, mp=True)

	env.reset(x_file_name=x_file_name)	
	max_n_step = p
	n_step = 0
	done = False

	while not done and n_step < max_n_step:
		# print(n_step)
		done = env.step()
		n_step += 1


if __name__ == '__main__':
	n_cpu = 10
	p = 50
	batch_n = 0

	model_scope = 'range'
	reward_format = 'binary'
	model_name = f'mp_{reward_format}_{model_scope}'

	log_p = int(np.log10(int(p)))
	p_sub_dir = f'p{log_p}'

	data_dir = f'./synthetic_data/{p_sub_dir}/batch_{batch_n}'
	x_file_list = [f for f in os.listdir(data_dir) if re.match('x', f)]
	n_file_batches = int(len(x_file_list)/n_cpu)
	rec_dir = '/gpfs/home/tdekara/thesis/replay_buffer'

	for i in range(2): # (n_file_batches): # TODO: change after testing
		file_names = x_file_list[i*n_cpu:(i+1)*n_cpu]
		print(file_names)
		for L0 in [1e-2]: # , 1e-3, 1e-4]:
			for L2 in [1e-2]: # , 1e-3, 1e-4]:
				# Clear replay buffer records
				for f in os.listdir(rec_dir):
					os.remove(os.path.join(rec_dir, f))

				# Run episodes
				args = zip(file_names, repeat(L0), repeat(L2), repeat(p), \
					repeat(batch_n), repeat(model_name))
				print(args)
				with get_context("spawn").Pool(n_cpu) as pool:
					pool.starmap(run_1_ep, args)
				print('done')
				# Combine action records
				record_list = [f for f in os.listdir(rec_dir)]
				array_list = []
				for file_name in record_list:
					record = np.load(f'{rec_dir}/{file_name}')
					array_list.append(record)

				record = np.vstack(array_list)
				print(f'rec_shape:{record.shape}')
				# Train Q model
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

				model = tf.keras.models.load_model(f'/gpfs/home/tdekara/thesis/models/{model_name}')
				log_dir = f"/gpfs/home/tdekara/thesis/tb_logs/{model_name}"
				os.makedirs(log_dir, exist_ok=True)
				tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

				model.fit(x, y, epochs=3, verbose=0, sample_weight=weights, \
					callbacks=[tensorboard_callback])

				model.save(f'/gpfs/home/tdekara/thesis/models/{model_name}')





