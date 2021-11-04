# So far, the script creates a 'BNBTree' object (as defined in l0bnb),
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

x_file_list = subprocess.run("cd synthetic_data; ls X* -1U", capture_output=True, text=True, shell=True).stdout.splitlines()
y_file_list = subprocess.run("cd synthetic_data; ls y* -1U", capture_output=True, text=True, shell=True).stdout.splitlines()
# b_file_list = subprocess.run("cd synthetic_data; ls b* -1U", capture_output=True, text=True, shell=True).stdout.splitlines()

l0=0.00001
l2=1

# class rl_env
# self.reset(l0, l2)
# load synthetic data
x_file_name = choice(x_file_list)
y_file_name = x_file_name.replace('X', 'y')
x = np.load(os.path.join('synthetic_data', x_file_name))
y = np.load(os.path.join('synthetic_data', y_file_name))
p = x.shape[1]
t = tree(x, y)
m=5

# TODO: look into the following (from 'tree.solve') ...
# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"

# initialize root node (from 'tree.solve')
t.root = Node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
t.number_of_nodes = 1

# Solve relaxation of root node
# This returns the primal value and dual value, and it updates
# self.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
relax_sol = t.root.lower_solve(l0, l2, m, solver='l1cd', rel_tol=1e-4, mio_gap=0)

# print(t.root.primal_beta[:5], t.root.support[:5])
# print(f'relax_sol: {relax_sol}')

#### Gather stats for 'observation'
# TODO: probably split this into a function??
# Note: 'i' will index the variables, and 'j' will index the nodes

# TODO: gather active indexes, 
# ie nodes that are active and variables active in those nodes
i = 3
node_j = t.root

### Stats for all x
# Note: 'cov_max' is the maximum other than the 1's along the diagonal
# TODO: maybe add some more quantiles
cov = x.shape[0] * np.cov(x, rowvar=False, bias=True)
cov_max = np.partition(cov.flatten(), kth = -(p+1))[-(p+1)] 
cov_min = cov.min()
cov_mean = cov.mean()
all_x_dot_y = np.matmul(x.T, y)

all_x_stats = [cov_min, cov_max, cov_mean,\
	all_x_dot_y.min(), all_x_dot_y.max(), all_x_dot_y.mean() ]

### Stats for x_i 
# Note: 'x_cov_max' is the max covariance other than the 1 for cov with itself
x_cov_min = cov[i].min()
x_cov_max = np.partition(cov[i], kth=-2)[-2]
x_cov_mean = cov[i].mean()
x_dot_y = np.dot(x[:,i],y)

x_i_stats = [x_cov_min, x_cov_max, x_cov_mean, x_dot_y]

# TODO: ### Stats for all nodes
# all_node_stats = []

# TODO: ### Stats for node_j
# node_j_stats = []

### Stats for x_i and node_j interaction
lb = 1 if i in node_j.zlb else 0
ub = 0 if i in node_j.zub else 1
# Note: len(node.primal_beta) == len(support)
if i in node_j.support:
	beta = node_j.primal_beta[node_j.support.index(i)]
else:
	beta = 0

x_i_node_j_stats = [lb, ub, beta]

### Collect all the stats 
# observation = [l0, l2, p, all_x_stats,  x_i_stats, all_node_stats, node_j_stats, x_i_node_j_stats]
# return(observation)

# self.step(action)

# return(observation, reward, done, info)
# TODO: handle branching
