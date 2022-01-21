import numpy as np
import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Note re: TF_CPP_MIN_LOG_LEVEL
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed 

def reverse_lookup(d, val):
  for key in d:
    if d[key] == val:
      return key

def get_q_hats(model_name, action_stats,  static_stats, epoch_n):
	model_path = f'./model_copies/epoch_{epoch_n}/{model_name}'
	model = tf.keras.models.load_model(model_path)

	# When getting q hats for all branhcing or all searching options, action_stats is a 2 dim
	# array where each row represents an available action.
	# When getting a single q hat (because of epsilon greedy policy), action_stats is a 1 dim array.
	if action_stats.ndim == 1:
		action_stats.shape = (1,-1)
	n_stats = action_stats.shape[0]
	model_input = np.hstack((np.tile(static_stats, (n_stats,1)), action_stats))
	model_input = tf.constant(model_input)
	q_hats = model.predict(model_input)
	
	return(q_hats)


def get_search_solution(node, p, l0, l2, y, epoch_n):
	# node.x_sub_mat will be the submatrix of the original matrix x, that excludes
	# the variables that have been branched down in the given node.  
	# node.x_sub_mat will also be rearranged so that the variables that have been 
	# branched up appear in the left-most columns.      
	node_active_x_i = []
	for k in range(p):
		if k not in node.zlb and k not in node.zub:
			 node_active_x_i.append(k)
		 
	# The first row of x_sub_mat has the indexes of the variables.
	x_sub_mat = np.hstack((node.x[:, node.zlb], node.x[:, node_active_x_i]))
	x_indexes = np.array((node.zlb + node_active_x_i), ndmin=2)
	x_sub_mat_indexed = np.vstack((x_indexes, x_sub_mat))

	path = f'./param_for_search/epoch_{epoch_n}'
	fname=f'{path}/x_sub_mat.csv'
	np.savetxt(fname=fname, X=x_sub_mat_indexed, fmt='%.18f', delimiter=',') 
	fname=f'{path}/lambdas.csv'
	np.savetxt(fname=fname,	X=np.array([l0, l2]), fmt='%.18f', delimiter=',') 
	fname=f'{path}/len_zub.csv'
	np.savetxt(fname=fname,	X=np.array([len(node.zub)]), fmt='%i', delimiter=',') 
	fname=f'{path}/y.csv'
	np.savetxt(fname=fname,	X=y, fmt='%.18f', delimiter=',')
 
	subprocess.run(f'Rscript search_script.R {epoch_n}', shell=True)

	path = f'./results_of_search/epoch_{epoch_n}'
	fname=f'{path}/support.csv'
	search_support = np.loadtxt(fname, delimiter=',', dtype='int', ndmin=1)
	fname=f'{path}/betas.csv'
	search_betas = np.loadtxt(fname, delimiter=',', ndmin=1)

	return(search_support, search_betas)

def int_sol(node, p, int_tol=10**-4, m=5):
	# Simple verion used for testing
	# if len(node.zlb + node.zub) == p:
	#	return(True)
	# else: 
	#	return(False)

	for i in node.support:
		if i not in node.zlb and i not in node.zub:
			# Note: the primal_beta values are indexed relative to len(support), 
			# and the new variable 'beta_i' is indexed relative to  p
			beta_i = node.primal_beta[node.support.index(i)]
			z_i = np.absolute(beta_i) / m
			residual = min(z_i, 1-z_i)
			if residual > int_tol:
				return(False)
	return(True)

	
def prin(**kwargs):
    for name, value in kwargs.items():
        print(name,': ', value)







		
