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
from l0bnb import BNBTree
from random import choice
from functions import get_static_stats
from functions import get_active_node_stats
from functions import get_action_specific_stats

m=5 
# m=5 works well for our synthetic data, but will need to adjusted for other data sets.

class rl_env:
	def __init__(self, l0=10**-5, l2=1, p=10**3):
		self.l0 = l0
		self.l2 = l2
		self.p = p
		self.x_file_list = subprocess.run( \
			f"cd synthetic_data; ls x*_p{int(np.log10(self.p))}_* -1U", \
			capture_output=True, text=True, shell=True).stdout.splitlines()

	def reset(self):
		# Load synthetic data
		x_file_name = np.random.choice(self.x_file_list)
		y_file_name = x_file_name.replace('x', 'y')
		b_file_name = x_file_name.replace('x', 'b')
		x = np.load(os.path.join('synthetic_data', x_file_name))
		y = np.load(os.path.join('synthetic_data', y_file_name))
		b = np.load(os.path.join('synthetic_data', b_file_name))
		global_stats = np.array([self.l0, self.l2, self.p])
		
		# Initialize tree
		t = BNBTree(x, y)
		# Initialize root node (from 'tree.solve')
		t.root = Node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		active_nodes = [t.root]
		active_x_i = list(range(self.p))

		# Solve relaxation of root node
		# 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		sol = t.root.lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		# for testing
		# test_node = Node(parent=t.root, zlb=[1,3,5], zub=[8,5])
		# active_nodes.append(test_node)

		# test_node.lower_solve(l0, l2, m, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		#### Gather stats for 'observation'
		cov = x.shape[0] * np.cov(x, rowvar=False, bias=True)
		
		static_stats = get_static_stats(cov, x, y, active_nodes, active_x_i, global_stats)
		
		### Stats for specific nodes and x_i
		action_specific_stats = get_action_specific_stats(active_nodes, self.p, cov, x, y)

		return([static_stats, action_specific_stats])

	# def self.step(action)


	# TODO: handle branching


	# active_x_i = []
	# for node in active_nodes:
	# 	active_x_i = active_x_i + [k for k in range(p) if k not in (node.zlb + node.zub)]

	# TODO: look into the following (from 'tree.solve') ...
	# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"

	# return(observation, reward, done, info)
