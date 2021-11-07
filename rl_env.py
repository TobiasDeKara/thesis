# So far, the script randomly selects data from the test bed,
# creates a 'BNBTree' object (as defined in l0bnb),
# initializes the root node using the 'Node' class (also from l0bnb),
# solves the integer relaxation of the root node using 'lower_solve' (also from l0bnb),
# and gathers statistics of the initial 'observation' (work in progress).
# 
# Much of the code below is taken directly from l0bnb, 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import os
import subprocess
import numpy as np
from l0bnb.node import Node
from l0bnb import BNBTree as tree
from random import choice
from functions import get_cov_percentiles
from functions import get_active_node_stats

x_file_list = subprocess.run("cd synthetic_data; ls X* -1U", capture_output=True, text=True, shell=True).stdout.splitlines()

l0=0.00001
l2=1
m=5

# class rl_env
# self.reset(l0, l2)
# load synthetic data
x_file_name = choice(x_file_list)
y_file_name = x_file_name.replace('X', 'y')
x = np.load(os.path.join('synthetic_data', x_file_name))
y = np.load(os.path.join('synthetic_data', y_file_name))
p = x.shape[1]
global_stats = np.array([l0, l2, p])
t = tree(x, y)

# TODO: look into the following (from 'tree.solve') ...
# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"

# initialize root node (from 'tree.solve')
t.root = Node(parent=None, zlb=[1], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
active_nodes = [t.root]
active_x_i = []
for node in active_nodes:
	active_x_i = active_x_i + [k for k in range(p) if k not in (node.zlb + node.zub)]

# Solve relaxation of root node
# This returns the primal value and dual value, and it updates
# self.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
relax_sol = t.root.lower_solve(l0, l2, m, solver='l1cd', rel_tol=1e-4, mio_gap=0)

# for testing
test_node = Node(parent=t.root, zlb=[1,3,5], zub=[8,5])
active_nodes.append(test_node)

test_node.lower_solve(l0, l2, m, solver='l1cd', rel_tol=1e-4, mio_gap=0)

#### Gather stats for 'observation'
cov = x.shape[0] * np.cov(x, rowvar=False, bias=True)
all_x_dot_y = np.matmul(x.T, y)

### Stats for all x
# Note: the cov_percentiles exclude the 1's from the diagonal
all_x_stats = np.append(get_cov_percentiles(cov),\
	np.quantile(all_x_dot_y,[0,0.25,0.5,0.75,1]))

### Stats for all ACTIVE x
cov_active = cov[active_x_i,:][:,active_x_i]
active_x_dot_y = all_x_dot_y[active_x_i]
active_x_stats = np.append(get_cov_percentiles(cov_active), \
	np.quantile(active_x_dot_y,[0,0.25,0.5,0.75,1]))

### Stats for all ACTIVE nodes
active_node_stats = get_active_node_stats(active_nodes)

counter = 0
### Stats for node
for node in active_nodes:
	len_support = len(node.support) if node.support else 0
	node_stats = np.array([len(node.zub), len(node.zlb), node.primal_value, len_support])
	
	# Find the x_i active in this specific node	
	node_active_x_i = [i for i in range(p) if i not in node.zlb and i not in node.zub] 
	
	### Stats for x_i
	for i in node_active_x_i:
		x_i_cov = np.partition(cov[i], -1)[:-1]
		x_i_cov_percentiles = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
		x_dot_y = np.dot(x[:,i], y)
		x_i_stats = np.append(x_i_cov_percentiles, x_dot_y)
	
		### Stats for x_i and node interaction
		lb = 1 if i in node.zlb else 0
		ub = 0 if i in node.zub else 1
		# Note: len(node.primal_beta) == len(support)
		if node.support and i in node.support:
			beta = node.primal_beta[node.support.index(i)]
		else:
			beta = 0
		x_i_node_stats = np.array([lb, ub, beta])

		### Collect all the stats
		arrays = [global_stats, all_x_stats, active_x_stats, active_node_stats, \
			node_stats, x_i_stats, x_i_node_stats] 
		observation_row = np.concatenate(arrays)
		
# return(observation)

# self.step(action)

# return(observation, reward, done, info)
# TODO: handle branching
