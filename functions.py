import numpy as np
import os
import subprocess
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed 


def get_q_hats(model_name, stats,  static_stats): 
	model = tf.keras.models.load_model(os.path.join('./models/', model_name))
	n_stats = stats.shape[0]
	model_input = np.hstack((np.tile(static_stats, (n_stats,1)), stats))
	model_input = tf.constant(model_input)
	q_hats = model.predict(model_input)
	
	return(q_hats)


def get_search_solution(node, p, l0, l2, y):
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

	np.savetxt(fname=os.path.join('./param_for_search', 'x_sub_mat.csv'), \
		X=x_sub_mat_indexed, fmt='%.18f', delimiter=',') 
	np.savetxt(fname=os.path.join('./param_for_search', 'lambdas.csv'), \
		X=np.array([l0, l2]), fmt='%.18f', delimiter=',') 
	np.savetxt(fname=os.path.join('./param_for_search', 'len_zub.csv'), \
		X=np.array([len(node.zub)]), fmt='%i', delimiter=',') 
	np.savetxt(fname=os.path.join('./param_for_search', 'y.csv'), \
		X=y, fmt='%.18f', delimiter=',')
 
	subprocess.run('Rscript search_script.R', shell=True)

	search_support = np.loadtxt(os.path.join('./results_of_search', 'support.csv'), \
		delimiter=',', dtype='int', ndmin=1)
	search_betas = np.loadtxt(os.path.join('./results_of_search', 'betas.csv'), delimiter=',', \
		ndmin=1)

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







		
