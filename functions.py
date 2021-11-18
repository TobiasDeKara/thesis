import numpy as np

def get_active_node_stats(active_nodes):
	len_zubs, len_zlbs, primal_values, len_supports = ([],)*4
	for key in active_nodes:
		node = active_nodes[key]
		len_zubs.append(len(node.zub))
		len_zlbs.append(len(node.zlb))
		primal_values.append(node.primal_value)
		len_supports.append(len(node.support))
	len_zubs_percentiles = np.quantile(len_zubs,[0,0.25,0.5,0.75,1])
	len_zlbs_percentiles = np.quantile(len_zlbs,[0,0.25,0.5,0.75,1])
	primal_values_percentiles = np.quantile(primal_values,[0,0.25,0.5,0.75,1])
	len_supports_percentiles = np.quantile(len_supports,[0,0.25,0.5,0.75,1])
	stats = (len_zubs_percentiles, len_zlbs_percentiles,\
		primal_values_percentiles, len_supports_percentiles)
	active_node_stats = np.concatenate(stats)
	active_node_sats = np.append(active_node_stats, len(active_nodes))
	return(active_node_stats)

def get_static_stats(cov, x, y, active_nodes, active_x_i, global_stats):
	all_x_dot_y = np.matmul(x.T, y)

	### Stats for all x
	# Note: the cov_percentiles exclude the 1's from the diagonal
	q = cov.shape[0]
	# flatten and remove the 1's from the diagonal
	cov_flat = np.partition(cov.flatten(), kth=-q)[:-q]
	all_x_stats = np.append(np.quantile(cov_flat, [0, 0.25, 0.5, 0.75, 1]), \
		np.quantile(all_x_dot_y,[0,0.25,0.5,0.75,1]))

	### Stats for all ACTIVE x
	active_x_dot_y = all_x_dot_y[active_x_i]
	if len(active_x_i) > 1:
		cov_active = cov[active_x_i,:][:,active_x_i]
		q = cov_active.shape[0]
		cov_active_flat = np.partition(cov_active.flatten(), -q)[:-q]
		active_x_stats = np.append(np.quantile(cov_active_flat,[0,0.25,0.5,0.75,1]), \
			np.quantile(active_x_dot_y,[0,0.25,0.5,0.75,1]))
	else:
		active_x_stats = np.append(np.zeros(5), \
			np.quantile(active_x_dot_y,[0,0.25,0.5,0.75,1]))
	
	active_x_stats = np.append(active_x_stats, len(active_x_i))

	### Stats for all ACTIVE nodes
	active_node_stats = get_active_node_stats(active_nodes)

	### Gather static stats
	static_stat_arrays = [global_stats, all_x_stats, active_x_stats, active_node_stats]
	static_stats = np.concatenate(static_stat_arrays)
	return(static_stats)

def get_action_specific_stats(active_nodes, p, cov, x, y):
	### Stats for node
	all_action_specific_stats = []
	for key in active_nodes:
		node = active_nodes[key]
		len_support = len(node.support) if node.support else 0
		node_stats = np.array([len(node.zub), len(node.zlb), node.primal_value, len_support])

		# Find the x_i active in this specific node     
		node_active_x_i = [i for i in range(p) if i not in node.zlb and i not in node.zub]

		### Stats for x_i
		for i in node_active_x_i:
			x_i_cov = np.partition(cov[i,:], -1)[:-1]
			x_i_cov_percentiles = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
			x_dot_y = np.dot(x[:,i], y)
			x_i_stats = np.append(x_i_cov_percentiles, x_dot_y)

			### Stats for x_i and node interaction
			lb = 1 if i in node.zlb else 0
			ub = 0 if i in node.zub else 1
			# Note: len(node.primal_beta) == len(support) != p, so
			# the beta values are indexed relative to len(support)
			if node.support and i in node.support:
				beta = node.primal_beta[node.support.index(i)]
			else:
				beta = 0
			x_i_node_stats = np.array([lb, ub, beta, i, key])

			### Collect action-specific stats
			action_specific_arrays = [node_stats, x_i_stats, x_i_node_stats]
			action_specific_stats = np.concatenate(action_specific_arrays)
			all_action_specific_stats.append(action_specific_stats)
	
	all_action_specific_stats = np.vstack(all_action_specific_stats)
	return(all_action_specific_stats)

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







		












	
