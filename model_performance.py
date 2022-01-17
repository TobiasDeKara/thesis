import numpy as np
import subprocess
import os

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
      
    # np.quantile(q_hat, np.linspace(0, 1, 10))
    
  return(f'branch_mse: {branch_mse}, search_mse: {search_mse}, \
  n_branches: {n_branches}, n_searches: {n_searches}')

# For reference
# model_record = np.array([
#     action.step_number, action.n_branch, action.n_search, \
#     (total_n_steps - action.step_number), \
#     (total_n_branch - action.n_branch), \
#     (total_n_search - action.n_search), \
#     action.q_hat[0], \
#     action.change_in_opt_gap, \
#     model_epoch])

# Mean Squared Error
# def get_mse(model_rec_file_list):
