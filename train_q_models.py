import os
import numpy as np
import subprocess
import tensorflow as tf

# A script to train the branch model and the search model on the action records
# found in './action_records/'.  After training, this script moves the
# action records to a new sub-directory './action_records/epoch_<n>', where <n>
# will be found using the number of sub-directories in './action_records/'.

branch_model_name = 'branch_model_in60_lay2'
branch_model = tf.keras.models.load_model(os.path.join('./models/', branch_model_name))
branch_record_list = subprocess.run(f'cd action_records; ls branch* -1U', capture_output=True, \
	text=True, shell=True).stdout.splitlines()

for branch_file_name in branch_record_list:
	branch_record = np.load(os.path.join('./action_records/', branch_file_name))
	n_col = branch_record.shape[1]
	x, q_hats, y = np.hsplit(branch_record, np.array([n_col-2, n_col-1]))
	branch_model.fit(x, y, epochs=10)

branch_model.save(os.path.join('./models/', branch_model_name))


search_model_name='search_model_in51_lay2'
search_model = tf.keras.models.load_model(os.path.join('./models/', search_model_name))
search_record_list = subprocess.run(f'cd action_records; ls search* -1U', capture_output=True, \
	text=True, shell=True).stdout.splitlines()

for search_file_name in search_record_list:
	search_record = np.load(os.path.join('./action_records/', search_file_name))
	n_col = search_record.shape[1]
	x, q_hats, y = np.hsplit(search_record, np.array([n_col-2, n_col-1]))
	search_model.fit(x, y, epochs=10)

search_model.save(os.path.join('./models/', search_model_name))

n_prev_epoch = subprocess.run('cd action_records; ls -d1U */ | wc -l', capture_output=True, \
	text=True, shell=True).stdout

epoch_n = int(n_prev_epoch) + 1

subprocess.run(f'cd action_records; mkdir epoch_{epoch_n}; mv *.npy epoch_{epoch_n}', shell=True)


