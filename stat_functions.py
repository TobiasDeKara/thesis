import numpy as np
import random

def get_active_node_stats(active_nodes):
	# Summary statisitics that describe the set of all active nodes
	# (The function that gets summary stats for each node is 'get_node_stats')
	len_zubs, len_zlbs, primal_values, len_supports, searched = [], [], [], [], []
	for key in active_nodes:
		node = active_nodes[key]
		len_zubs.append(len(node.zub))
		len_zlbs.append(len(node.zlb))
		primal_values.append(node.primal_value)
		len_supports.append(len(node.support))
		searched.append(node.searched)
	len_zubs_percentiles = np.quantile(len_zubs,[0,0.25,0.5,0.75,1])
	len_zlbs_percentiles = np.quantile(len_zlbs,[0,0.25,0.5,0.75,1])
	primal_values_percentiles = np.quantile(primal_values,[0,0.25,0.5,0.75,1])
	len_supports_percentiles = np.quantile(len_supports,[0,0.25,0.5,0.75,1])
	mean_searched = np.mean(searched, keepdims=True)
	stats = (len_zubs_percentiles, len_zlbs_percentiles,\
		primal_values_percentiles, len_supports_percentiles, mean_searched)
	active_node_stats = np.concatenate(stats)
	active_node_stats = np.append(active_node_stats, len(active_nodes))
	return(active_node_stats)

def get_node_stats(node, node_key, lower_bound_node_key, upper_bound_node_key):
	# Returns 5 summary stats for a given node
	len_support = len(node.support) if node.support else 0
	has_lb = (lower_bound_node_key == node_key)
	has_ub = (upper_bound_node_key == node_key)
	node_stats = np.array([len(node.zub), len(node.zlb), node.primal_value, \
	len_support, node.searched, has_lb, has_ub], dtype=float, ndmin=1)
	return(node_stats)

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
	# static_stats.shape = (static_stats.shape[0], 1)
	return(static_stats)


def get_all_action_stats(active_nodes, p, cov, x, y, lower_bound_node_key, upper_bound_node_key):
	# Note: the 'searching_stats' are the same as 'node_stats' because for searching we only need to chose a node to search in
	# Returns:
	# 	1.  'all_branching_stats': an np.array of stats for branching, one row per active node/x_i pair,
	# 	2.  'all_searching_stats': an np.array of stats for searching, one row per active node
	# 	3.  'all_branching_keys': an np.array, first column is the variable index, 2nd col is node key 
	# 	4.  'all_searching_keys': an np.array of strings of node keys

	all_branching_stats, all_branching_keys, all_searching_stats, all_searching_keys = [], [], [], []

	for key in active_nodes:
		### Stats for node
		node = active_nodes[key]
		node_stats = get_node_stats(node, key, lower_bound_node_key, upper_bound_node_key)
		all_searching_stats.append(node_stats)
		all_searching_keys.append(key)

		# Find the x_i active in this specific node     
		node_active_x_i = [i for i in range(p) if i not in node.zlb and i not in node.zub]

		for i in node_active_x_i:
			### Stats for x_i
			x_i_cov = np.partition(cov[i,:], -1)[:-1]
			x_i_cov_percentiles = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
			x_dot_y = np.dot(x[:,i], y)

			# Note: len(node.primal_beta) == len(support) != p, so
			# the beta values are indexed relative to len(support)
			if node.support and i in node.support:
				x_in_node_support = 1
				beta = node.primal_beta[node.support.index(i)]
			else:
				x_in_node_support = 0
				beta = 0

			x_i_stats = np.concatenate((x_i_cov_percentiles, \
				np.array([x_dot_y, x_in_node_support, beta])))

			branching_keys = np.array([i, key], dtype=str)
			all_branching_keys.append(branching_keys)

			### Collect branching-specific stats
			branching_specific_arrays = [node_stats, x_i_stats]
			branching_specific_stats = np.concatenate(branching_specific_arrays)
			all_branching_stats.append(branching_specific_stats)
        
	# Convert results to np.arrays instead of lists (because this is the format used by OpenAI's gym package)
	all_branching_stats = np.vstack(all_branching_stats)
	all_searching_stats = np.vstack(all_searching_stats)
	all_branching_keys = np.vstack(all_branching_keys)
	all_searching_keys = np.array(all_searching_keys, dtype=str)
	
	return(all_branching_stats, all_searching_stats, all_branching_keys, all_searching_keys)


def get_random_action_stats(active_nodes, p, cov, x, y, lower_bound_node_key, upper_bound_node_key):
	# Note: the 'search_stats' are the same as 'node_stats' because for searching we only need to chose a node to search in
	# Returns:
	# 	1.  'branch_stats': an np vector of stats for branching on a randomly chosen active x_i in the randomly chosen active node,
	# 	2.  'search_stats': an np vector of stats of a randomly chosen active node (not necessarily the one chosen for branching)
	#	3.  'branch_keys': an np vector of strings, the x_i index and the node key
	# 	4.  'search_key': a string, the node key for the randomly chosen search node

	### Search
	search_key, search_node = random.choice(list(active_nodes.items()))
	search_key = np.array(search_key, dtype=str, ndmin=1)
	search_stats = get_node_stats(search_node, search_key, lower_bound_node_key, upper_bound_node_key)
	
	### Branch
	branch_node_key, branch_node = random.choice(list(active_nodes.items()))
	branch_node_stats = get_node_stats(branch_node, branch_node_key, \
	lower_bound_node_key, upper_bound_node_key)

	# Randomly choose an x_i that is active in this specific node     
	node_active_x_i = [i for i in range(p) if i not in branch_node.zlb and i not in branch_node.zub]
	i = random.choice(node_active_x_i)

	# Stats for x_i
	x_i_cov = np.partition(cov[i,:], -1)[:-1]
	x_i_cov_percentiles = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
	x_dot_y = np.dot(x[:,i], y)

	# Note: the use of '.index(i)' below is becuase 
	# len(node.primal_beta) == len(support) != p,
	# so the beta values are indexed relative to len(support)
	if branch_node.support and i in branch_node.support:
		x_in_node_support = 1
		node_beta = branch_node.primal_beta[branch_node.support.index(i)]
	else:
		x_in_node_support = 0
		node_beta = 0

	x_i_stats = np.concatenate((x_i_cov_percentiles, np.array([x_dot_y, x_in_node_support, node_beta])))

	# Collect branch stats
	branch_keys = np.array([i, branch_node_key], dtype=str, ndmin=2)
	branch_arrays = [branch_node_stats, x_i_stats]
	branch_stats = np.concatenate(branch_arrays)
	branch_stats = np.array(branch_stats, ndmin=2)

	return(branch_stats, search_stats, branch_keys, search_key)

