import os
import numpy as np
import subprocess
import tensorflow as tf

# A script to run another epoch of training by using all of the
# action records found in './action_records/used'.

branch_model_name = 'branch_model_in58_lay2_0'
branch_model = tf.keras.models.load_model(os.path.join('./models/', branch_model_name))
branch_record_list = subprocess.run(f'cd action_records/used; ls branch* -1U', capture_output=True, \
	text=True, shell=True).stdout.splitlines()

for branch_file_name in branch_record_list:
	branch_record = np.load(os.path.join('./action_records/used/', branch_file_name))
	n_col = branch_record.shape[1]
	x, q_hats, y = np.hsplit(branch_record, np.array([n_col-2, n_col-1]))
	branch_model.fit(x, y, epochs=10)

branch_model.save(os.path.join('./models/', branch_model_name))


search_model_name='search_model_in49_lay2_0'
search_model = tf.keras.models.load_model(os.path.join('./models/', search_model_name))
search_record_list = subprocess.run(f'cd action_records/used; ls search* -1U', capture_output=True, \
	text=True, shell=True).stdout.splitlines()

for search_file_name in search_record_list:
	search_record = np.load(os.path.join('./action_records/used/', search_file_name))
	n_col = search_record.shape[1]
	x, q_hats, y = np.hsplit(search_record, np.array([n_col-2, n_col-1]))
	search_model.fit(x, y, epochs=10)

search_model.save(os.path.join('./models/', search_model_name))

# subprocess.run('cd action_records; mv *.npy used', shell=True)

