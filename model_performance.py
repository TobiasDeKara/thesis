import numpy as np
import pandas as pd
import subprocess
import os
import tensorflow as tf
# Note: model_record format is at bottom for reference

def get_run_stats(run_n=0, validation=False):
  for i in range(2):
    sum_sq = 0
    n_obs = 0
    sum_obs = 0
    sum_pred = 0
    n_non_zero_obs = 0
    n_non_zero_pred = 0

    if i == 0: # branch
      rec_file_name = f'./combined_model_records/run_{run_n}/branch_mod_rec_comb.npy'
    else: # search
      rec_file_name = f'./combined_model_records/run_{run_n}/search_mod_rec_comb.npy'

    model_rec = np.load(rec_file_name)
    n_col = record.shape[1]
    x, y = np.hsplit(branch_record, np.array([n_col-1]))

    if validation:
      if i == 0: # branch
        model_name = 'branch_model_in61_lay2'
        record = np.load(f'./combined_action_records/run_{run_n}/branch_rec_comb.npy'
      else: # search
        model_name = 'search_model_in53_lay2'
        record = np.load(f'./combined_action_records/run_{run_n}/search_rec_comb.npy'


      q_hat = model_rec[:, model_rec.shape[1]-3]
    change_in_opt_gap = model_rec[:, model_rec.shape[1]-2]
    sum_sq = sum((q_hat - change_in_opt_gap)**2)
    n_obs = model_rec.shape[0]
    sum_obs = change_in_opt_gap.sum()
    sum_pred = q_hat.sum()
    n_non_zero_obs = (change_in_opt_gap > 0.0001).sum()
    n_non_zero_pred = (q_hat > 0.0001).sum()
   
    if i == 1: # search
      search_mse = sum_sq / n_obs
      n_searches = n_obs
      mean_search_obs = sum_obs / n_obs
      mean_search_pred = sum_pred / n_obs
      mean_non_zero_search_obs = sum_obs / n_non_zero_obs # Note sum of non-zero elements == sum of all elements
      mean_non_zero_search_pred = sum_pred / n_non_zero_pred
      n_non_zero_search_obs = n_non_zero_obs
      n_non_zero_search_pred = n_non_zero_pred
    else: # branch
      branch_mse = sum_sq / n_obs
      n_branches = n_obs
      mean_branch_obs = sum_obs / n_obs
      mean_branch_pred = sum_pred / n_obs
      mean_non_zero_branch_obs = sum_obs / n_non_zero_obs # Note sum of non-zero elements == sum of all elements
      mean_non_zero_branch_pred = sum_pred / n_non_zero_pred
      n_non_zero_branch_obs = n_non_zero_obs
      n_non_zero_branch_pred = n_non_zero_pred

  # Gather return values
  out = pd.DataFrame([[n_branches, n_searches], 
  [mean_branch_obs, mean_search_obs],
  [mean_branch_pred, mean_search_pred], 
  [branch_mse, search_mse],
  [mean_non_zero_branch_obs, mean_non_zero_search_obs], 
  [mean_non_zero_branch_pred, mean_non_zero_search_pred],
  [n_non_zero_branch_obs, n_non_zero_search_obs],
  [n_non_zero_branch_pred, n_non_zero_search_pred]],
  index = ['n_obs', 'mean_obs', 'mean_pred', 'mse', 'mean_non_zero_obs', \
  'mean_non_zero_pred', 'n_non_zero_obs', 'n_non_zero_pred'], 
  columns=['branch', 'search'])

  return(out)

def get_validation_run_stats(run_n=0):
    for i in range(2): 
        sum_sq = 0
        if i == 0:
            # branch
            branch_model_name = 'branch_model_in61_lay2'
            branch_model = tf.keras.models.load_model(f'./models/{branch_model_name}')


                branch_record = np.load(f'./action_records/run_{run_n}/{branch_file_name}')
                n_col = branch_record.shape[1]
                x, y = np.hsplit(branch_record, np.array([n_col-1]))
                batch_mse = branch_model.evaluate(x, y, verbose=0)
                batch_size =  branch_record.shape[0]
                sum_sq += batch_mse*batch_size
                n_branches += batch_size
            branch_mse = sum_sq / n_branches
        else:
            # Repeat for search
            search_model_name='search_model_in53_lay2'
            search_model = tf.keras.models.load_model(f'./models/{search_model_name}')
            search_model_name = ''
            search_record_list = subprocess.run(f'cd action_records/run_{run_n}; ls search* -1U | head -1000', \
            capture_output=True, text=True, shell=True).stdout.splitlines()

            n_searches = 0
            for search_file_name in search_record_list:
                search_record = np.load(f'./action_records/run_{run_n}/{search_file_name}')
                n_col = search_record.shape[1]
                x, y = np.hsplit(search_record, np.array([n_col-1]))
                batch_mse = search_model.evaluate(x, y, verbose=0)
                batch_size = search_record.shape[0]
                sum_sq += batch_mse*batch_size
                n_searches += batch_size
            search_mse = sum_sq / n_searches
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
#     model_batch])

