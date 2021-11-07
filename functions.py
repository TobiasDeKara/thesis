import numpy as np

def get_cov_percentiles(cov):
	q = cov.shape[0]
	# flatten and remove the 1's from the diagonal
	cov_flat = np.partition(cov.flatten(), kth=-q)[:-q] 
	return(np.quantile(cov_flat, [0, 0.25, 0.5, 0.75, 1]))

def get_active_node_stats(active_nodes):
	len_zubs = [len(node.zub) for node in active_nodes]
	len_zlbs = [len(node.zlb) for node in active_nodes]
	len_zubs_percentiles = np.quantile(len_zubs,[0,0.25,0.5,0.75,1])
	len_zlbs_percentiles = np.quantile(len_zlbs,[0,0.25,0.5,0.75,1])
	primal_values = [node.primal_value for node in active_nodes]
	primal_values_percentiles = np.quantile(primal_values,[0,0.25,0.5,0.75,1])
	len_supports = [len(node.support) for node in active_nodes]
	len_supports_percentiles = np.quantile(len_supports,[0,0.25,0.5,0.75,1])
	stats = (len_zubs_percentiles, len_zlbs_percentiles,\
		primal_values_percentiles, len_supports_percentiles)
	active_node_stats = np.concatenate(stats)
	return(active_node_stats)
