# This script implements a reinforcement learning environment,
# in the format of OpenAI's package 'gym'.

# Much of the code below is taken directly from 'l0bnb', 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

# self.reset() randomly selects data from the test bed for a given p,
# creates a 'BNBTree' object (as defined in 'l0bnb'),
# initializes the root node using the 'Node' class (also from 'l0bnb'),
# solves the integer relaxation of the root node using 'lower_solve' (also from 'l0bnb'),
# and gathers statistics of the initial 'observation'.
# 

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
		self.active_nodes = None
		self.node_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None

		# TODO: we don't need to calulate the cov_stats at every step, so let's attach them to the rl_env
		# self.cov_stats = None

	def reset(self):
		### Load synthetic data
		x_file_name = np.random.choice(self.x_file_list)
		y_file_name = x_file_name.replace('x', 'y')
		b_file_name = x_file_name.replace('x', 'b')
		self.x = np.load(os.path.join('synthetic_data', x_file_name))
		self.y = np.load(os.path.join('synthetic_data', y_file_name))
		self.b = np.load(os.path.join('synthetic_data', b_file_name))
		global_stats = np.array([self.l0, self.l2, self.p])
		
		### Initialize tree
		t = BNBTree(self.x, self.y)
		# Initialize root node (from 'tree.solve')
		t.root = Node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		self.active_nodes = [t.root]
		active_x_i = list(range(self.p))

		### Solve relaxation of root node
		# 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		t.root.lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		#### Gather stats for 'observation'
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)
		
		### Stats for specific nodes and x_i
		action_specific_stats = get_action_specific_stats( \
			self.active_nodes, self.p, self.cov, self.x, self.y)
		
		# Note: the last two columns of action_specific_stats have the x_i and node indexes
		# The x_i indexes are relative to p, the node indexes are relative to len(active_nodes)
		return([static_stats, action_specific_stats])

	def step(self, action):
		"""
		action: an x_i index and node index.  The x_i index is relative to p, and the node index
			is relative to len(active_nodes).	
		"""
		### Branch
		branching_x_i = action[0]
		branching_node = self.active_nodes[action[1]]
		new_zlb = branching_node.zlb
		new_zlb.append(branching_x_i)
		new_zub = branching_node.zub
		new_zub.append(branching_x_i)
		self.active_nodes.append( \
			Node(parent=branching_node, zlb=new_zlb, zub=branching_node.zub))
		self.active_nodes.append( \
			Node(parent=branching_node, zlb=branching_node.zlb, zub=new_zub))
		self.node_counter += 2
		del self.active_nodes[action[1]]
		
	
		### Solve relaxations in new nodes
		# TODO: look into the following (from 'tree.solve') ...
		# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"
		self.active_nodes[-1].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		self.active_nodes[-2].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		# TODO: check if integer solution, and if so, check if active nodes can be eliminated
	
		### Stats
		global_stats = np.array([self.l0, self.l2, self.p])		
		active_x_i = []
		for node in self.active_nodes:
			active_x_i = active_x_i + [k for k in range(self.p) if k not in (node.zlb + node.zub + active_x_i)]
			if len(active_x_i) == self.p:
				break
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)

                ### Stats for specific nodes and x_i
		action_specific_stats = get_action_specific_stats( \
			self.active_nodes, self.p, self.cov, self.x, self.y)

                # Note: the last two columns of action_specific_stats have the x_i and node indexes
                # The x_i indexes are relative to p, the node indexes are relative to len(active_nodes)
		return([static_stats, action_specific_stats])

		# return(observation, reward, done, info)
