import sys
import os
import numpy as np
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# A script to train the branch model and the search model on the action records
# found in './action_records/batch_<n>', where <n> is passed to this 
# script from the command line call of 'run_rl.py'.

batch_n = int(sys.argv[1])

# Update parameters of Branch Model
branch_model_name = 'branch_model_in60_lay2'
branch_model = tf.keras.models.load_model(f'./models/{branch_model_name}')
branch_record_list = subprocess.run(f'cd action_records/batch_{batch_n}; ls branch* -1U', \
	capture_output=True, text=True, shell=True).stdout.splitlines()

for branch_file_name in branch_record_list:
	branch_record = np.load(f'./action_records/batch_{batch_n}/{branch_file_name}')
	n_col = branch_record.shape[1]
	x, y = np.hsplit(branch_record, np.array([n_col-1]))
	branch_model.fit(x, y, epochs=3, verbose=0)

branch_model.save(f'./models/{branch_model_name}')

# Repeat for Search Model
search_model_name='search_model_in51_lay2'
search_model = tf.keras.models.load_model(f'./models/{search_model_name}')
search_record_list = subprocess.run(f'cd action_records/batch_{batch_n}; ls search* -1U', \
	capture_output=True, text=True, shell=True).stdout.splitlines()

for search_file_name in search_record_list:
	search_record = np.load(f'./action_records/batch_{batch_n}/{search_file_name}')
	n_col = search_record.shape[1]
	x, y = np.hsplit(search_record, np.array([n_col-1]))
	search_model.fit(x, y, epochs=3, verbose=0)

search_model.save(f'./models/{search_model_name}')
