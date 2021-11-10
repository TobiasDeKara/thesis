# This script implements a reinforcement learning environment,
# in the format of OpenAI's package 'gym'.

# Much of the code below is taken directly from 'l0bnb', 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]


import os
import subprocess
import numpy as np
from copy import copy
from l0bnb.node import Node
from l0bnb import BNBTree
from random import choice
from functions import get_static_stats, get_active_node_stats, get_action_specific_stats, int_sol

class rl_env:
	def __init__(self, l0=10**-5, l2=1, p=10**3, m=5, alpha=1):
		self.l0 = l0
		self.l2 = l2
		self.p = p
		self.x_file_list = subprocess.run( \
			f"cd synthetic_data; ls x*_p{int(np.log10(self.p))}_* -1U", \
			capture_output=True, text=True, shell=True).stdout.splitlines()
		self.int_tol = 10**-4
		# m=5 works well for our synthetic data, but will need to adjusted for other data sets.
		self.m = m	
		self.alpha = alpha

		# All of the following attributes are reset by 'reset_rl_evn_vars()' which
		# is called in 'reset(self)' below.
		self.active_nodes = None
		self.node_counter = 0
		self.search_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None
		self.curr_best_primal = None
		self.curr_best_beta = None
		self.curr_best_support = None

	def reset_rl_env_vars(self):
		self.active_nodes = None
		self.node_counter = 0
		self.search_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None
		self.curr_best_primal = None
		self.curr_best_beta = None
		self.curr_best_support = None

		# TODO: we don't need to calulate the cov_stats at every step, so let's attach them to the rl_env
		# self.cov_stats = None

	def reset(self):
		"""
		Creates the initial observation with a new x and y.
		Returns a list of two numpy arrays: 'static_stats' (44 by 1) and 'action_specific_stats' (p by 15)
		""" 
		self.reset_rl_env_vars()

		### Randomly select data from the test bed, with the given number of variables, 'self.p'
		x_file_name = np.random.choice(self.x_file_list)
		y_file_name = x_file_name.replace('x', 'y')
		b_file_name = x_file_name.replace('x', 'b')
		self.x = np.load(os.path.join('synthetic_data', x_file_name))
		self.y = np.load(os.path.join('synthetic_data', y_file_name))
		self.b = np.load(os.path.join('synthetic_data', b_file_name))
		global_stats = np.array([self.l0, self.l2, self.p])
		
		### Initialize a 'BNBTree' object (as defined in 'l0bnb'), and initialize its root node
		t = BNBTree(self.x, self.y)
		t.root = Node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		self.node_counter = 1
		self.active_nodes = [t.root]
		active_x_i = list(range(self.p))
		# Note: 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		t.root.lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		#### Gather stats for 'observation'
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)
		action_specific_stats = get_action_specific_stats( \
			self.active_nodes, self.p, self.cov, self.x, self.y)
		
		# Note: the last two columns of action_specific_stats have the x_i and node indexes;
		# the x_i indexes are relative to p, the node indexes are relative to len(active_nodes).
		return([static_stats, action_specific_stats])

	def step(self, action):
		"""
		action: an x_i index and node index.  The x_i index is relative to p, and the node index
			is relative to len(active_nodes).	
		"""
		### Branch
		branching_x_i = action[0]
		branching_node = self.active_nodes[action[1]]
		new_zlb = branching_node.zlb.copy()
		new_zlb.append(branching_x_i)
		new_zub = branching_node.zub.copy()
		new_zub.append(branching_x_i)
		self.active_nodes.append( \
			Node(parent=branching_node, zlb=new_zlb, zub=branching_node.zub))
		self.active_nodes.append( \
			Node(parent=branching_node, zlb=branching_node.zlb, zub=new_zub))
		self.node_counter += 2
		del self.active_nodes[action[1]] 
		# TODO: the line above only works if action[1] is non-negative, so let's add a test for that
		
	
		### Solve relaxations in new nodes
		# TODO: look into the following (from 'tree.solve') ...
		# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"
		self.active_nodes[-1].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		self.active_nodes[-2].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		# Check if either new node is an integer solution
		for node in [self.active_nodes[-2], self.active_nodes[-1]]: # This order matters if we remove the 1st one
			if int_sol(node, int_tol=self.int_tol, m=self.m):
				# del self.active_nodes[self.active_nodes.index(node)]
				self.active_nodes.remove(node)
				# Check if new integer solution is the best known integer solution
				if (not self.curr_best_primal) or (node.primal_value < self.curr_best_primal):
					self.curr_best_primal = node.primal_value
					self.curr_best_beta = node.primal_beta
					self.curr_best_support = node.support
					# Check if other nodes can be eliminated
					for other_node in self.active_nodes:
						if self.curr_best_primal < other_node.primal_value:
							del self.active_nodes[other_node]
							# Check if all nodes have been eliminated
							if len(self.active_nodes) == 0:
								### Gather return values
								done = True
								newline = '\n'
								observation = f'primal value: {self.curr_best_primal}{newline} beta: {self.curr_best_beta}{newline}support: {self.curr_best_support}'
								reward = -self.node_counter - self.alpha * self.search_counter
								info = f'node count: {self.node_counter}{newline} \
								search count: {self.search_counter}'
								return(observation, reward, done, info)

		### Stats
		global_stats = np.array([self.l0, self.l2, self.p])		
		active_x_i = []
		for node in self.active_nodes:
			active_x_i = active_x_i + \
				[k for k in range(self.p) if k not in node.zlb and k not in node.zub and k not in active_x_i]
			if len(active_x_i) == self.p:
				break

		### Gather stats for 'observation'
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)
		action_specific_stats = get_action_specific_stats( \
			self.active_nodes, self.p, self.cov, self.x, self.y)
                # Note: the last two columns of action_specific_stats have the x_i and node indexes
                # The x_i indexes are relative to p, the node indexes are relative to len(active_nodes)
		
		### Gather return values
		done = False
		observation = [static_stats, action_specific_stats]
		reward = 0
		# TODO: formating with newline is not working
		newline = '\n'
		info = f'node count: {self.node_counter}{newline} search count: {self.search_counter}'
		return(observation, reward, done, info)

		# return(observation, reward, done, info)
