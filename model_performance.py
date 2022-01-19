import numpy as np
import subprocess
import os
import tensorflow as tf
# Note: model_record format is at bottom for reference

# Mean squared error
def get_mse(epoch_n):
  for i in range(2):
    if i == 0:
      # search records
      model_rec_file_list = subprocess.run( \
      f"cd model_records/epoch_{epoch_n}; ls search_model_records* -1U", \
      capture_output=True, text=True, shell=True).stdout.splitlines()
    else:
      # branch records
      model_rec_file_list = subprocess.run( \
      f"cd model_records/epoch_{epoch_n}; ls branch_model_records* -1U", \
      capture_output=True, text=True, shell=True).stdout.splitlines()

    sum_sq = 0
    n_obs = 0
    for model_rec_file_name in model_rec_file_list:
      model_rec = np.load(os.path.join(f'model_records/epoch_{epoch_n}', model_rec_file_name))
      q_hat = model_rec[:, model_rec.shape[1]-3]
      change_in_opt_gap = model_rec[:, model_rec.shape[1]-2]
      sum_sq += sum((q_hat - change_in_opt_gap)**2)
      n_obs += model_rec.shape[0]
    
    if i == 0:
      search_mse = sum_sq / n_obs
      n_searches = n_obs
    else:
      branch_mse = sum_sq / n_obs
      n_branches = n_obs
      
    n_non_zero_obs = change_in_opt_gap

  return(f'branch_mse: {branch_mse}, search_mse: {search_mse}, \
  n_branches: {n_branches}, n_searches: {n_searches}')

def get_validation_mse(epoch_n=1):
    for i in range(2): 
        sum_sq = 0
        if i == 0:
            # branch
            branch_model_name = 'branch_model_in60_lay2'
            branch_model = tf.keras.models.load_model(f'./models/{branch_model_name}')
            branch_record_list = subprocess.run(f'cd action_records/epoch_{epoch_n}; ls branch* -1U', \
            capture_output=True, text=True, shell=True).stdout.splitlines()

            n_branches = 0
            for branch_file_name in branch_record_list:
                branch_record = np.load(f'./action_records/epoch_{epoch_n}/{branch_file_name}')
                n_col = branch_record.shape[1]
                x, y = np.hsplit(branch_record, np.array([n_col-1]))
                batch_mse = branch_model.evaluate(x, y, verbose=0)
                batch_n =  branch_record.shape[0]
                sum_sq += batch_mse*batch_n
                n_branches += batch_n
                branch_mse = sum_sq / n_branches
        else:
            # Repeat for search
            search_model_name='search_model_in51_lay2'
            search_model = tf.keras.models.load_model(f'./models/{search_model_name}')
            search_model_name = ''
            search_record_list = subprocess.run(f'cd action_records/epoch_{epoch_n}; ls search* -1U', \
            capture_output=True, text=True, shell=True).stdout.splitlines()

            n_searches = 0
            for search_file_name in search_record_list:
                search_record = np.load(f'./action_records/epoch_{epoch_n}/{search_file_name}')
                n_col = search_record.shape[1]
                x, y = np.hsplit(search_record, np.array([n_col-1]))
                batch_mse = search_model.evaluate(x, y, verbose=0)
                batch_n = search_record.shape[0]
                sum_sq += batch_mse*batch_n
                n_searches += batch_n
                search_mse = sum_sq / n_searches
    return(f'branch_mse: {branch_mse}, search_mse: {search_mse}, \
    n_branches: {n_branches}, n_searches: {n_searches}')

    # model_rec_file_list = subprocess.run( \
    # f"cd model_records/epoch_{epoch_n}; ls search_model_records* -1U", \
    # capture_output=True, text=True, shell=True).stdout.splitlines()
    
    # model_rec_file_name = model_rec_file_list[0]
    #  model_rec = np.load(os.path.join(f'model_records/epoch_{epoch_n}', model_rec_file_name))

# For reference
# model_record = np.array([
#     action.step_number, action.n_branch, action.n_search, \
#     (total_n_steps - action.step_number), \
#     (total_n_branch - action.n_branch), \
#     (total_n_search - action.n_search), \
#     action.q_hat[0], \
#     action.change_in_opt_gap, \
#     model_epoch])

