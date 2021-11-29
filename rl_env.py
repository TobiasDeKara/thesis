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
from functions import *
from stat_functions import *
import math

class rl_env:
	def __init__(self, l0=10**-5, l2=1, p=10**3, m=5, alpha=1, branch_model_name='branch_model_in58_lay2_0', \
			search_model_name='search_model_in49_lay2_0'):
		self.l0 = l0
		self.l2 = l2
		self.p = p
		self.branch_model_name = branch_model_name
		self.search_model_name = search_model_name

		# Changed for mini data (p=5)
		self.x_file_list = subprocess.run( \
			f"cd synthetic_data; ls x*_pmini_* -1U", \
			capture_output=True, text=True, shell=True).stdout.splitlines()
		
		#self.x_file_list = subprocess.run( \
		#	f"cd synthetic_data; ls x*_p{int(np.log10(self.p))}_* -1U", \
		#	capture_output=True, text=True, shell=True).stdout.splitlines()
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
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None

	def reset_rl_env_vars(self):
		self.active_nodes = dict()
		self.node_counter = 0
		self.search_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None

		# TODO: we don't need to calulate the cov_stats at every step, so let's attach them to the rl_env
		# self.cov_stats = None

	def reset(self):
		"""
		Creates the initial observation with a new x and y.
		Returns a list of two numpy arrays: 'static_stats' (44 by 1) and 'action_specific_stats' (p by 15)
		""" 
		self.reset_rl_env_vars()

		### Randomly select data from the test bed, with the given number of variables, 'self.p'
		# TODO: allow requests for particular data by seed value
		x_file_name = np.random.choice(self.x_file_list)
		y_file_name = x_file_name.replace('x', 'y')
		b_file_name = x_file_name.replace('x', 'b')
		self.x = np.load(os.path.join('synthetic_data', x_file_name))
		self.y = np.load(os.path.join('synthetic_data', y_file_name))
		self.b = np.load(os.path.join('synthetic_data', b_file_name))
		global_stats = np.array([self.l0, self.l2, self.p], dtype=float)
		
		### Initialize a 'BNBTree' object (as defined in 'l0bnb'), and initialize its root node
		t = BNBTree(self.x, self.y)
		t.root = Node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		self.active_nodes['root_node'] = t.root
		active_x_i = list(range(self.p))
		# Note: 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		t.root.lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		#### Gather stats for 'observation'
		# Note: 'action_keys' (3rd line below)  is an np.array of strings that has the x_i indexes and node keys;
		# the x_i indexes are relative to p.
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)
		branch_stats, search_stats, branch_keys, search_keys = \
			get_action_specific_stats(self.active_nodes, self.p, self.cov, self.x, self.y)

		# Get q_hats by applying models
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, branch_stats.shape[0], static_stats)
		search_q_hats = get_q_hats(self.search_model_name, search_stats, search_stats.shape[0], static_stats)

		### Gather return values
		action_q_hats = np.concatenate((branch_q_hats, search_q_hats))
		action_keys  =  np.concatenate((branch_keys, search_keys))

		return([action_q_hats, action_keys])

	def eliminate_nodes(self, node_key):
		node = self.active_nodes[node_key]
		
		# Can the node be eliminated?
		if self.curr_best_int_primal < node.primal_value:
			del self.active_nodes[node_key]

		# Is the  new node an integer solution?
		# If so, then it is the new best integer solution, because otherwise it would have been
		# eliminated in the step above. 
		elif int_sol(node, p=self.p, int_tol=self.int_tol, m=self.m):
			# prin(int_sol_node_primal=node.primal_value)
			# prin(curr_best_before=self.curr_best_int_primal)

			# TODO: no 'copy()' after 'node.primal_value' in next line?
			self.curr_best_int_primal = node.primal_value
			# prin(curr_best_after=self.curr_best_int_primal)
			self.curr_best_int_beta = node.primal_beta.copy()
			self.curr_best_int_support = node.support.copy()
			del self.active_nodes[node_key]

			# Can other nodes be eliminated?
			for other_key in list(self.active_nodes.keys()):
				if self.active_nodes[other_key].primal_value > self.curr_best_int_primal:
					# prin(other_node_primal=self.active_nodes[other_key].primal_value)
					del self.active_nodes[other_key]
					# print(f'removing key: {other_key}')

# TODO: in search routine set intercept=FALSE

	def eliminate_after_search(self, search_betas, search_support):
		residuals = self.y - np.matmul(self.x[:,search_support], search_betas)
		rss = np.dot(residuals, residuals)
		search_sol_primal = rss/2 + self.l0*len(search_support) + self.l2*np.dot(search_betas, search_betas)
		if search_sol_primal < self.curr_best_int_primal:
			prin(search_sol_primal=search_sol_primal)
			prin(curr_best_int_primal=self.curr_best_int_primal)
			# Update current best integer solution
			self.curr_best_int_primal = search_sol_primal.copy()
			self.curr_best_int_beta = search_betas.copy()
			self.curr_best_int_support = search_support.copy()

			# Can other nodes be eliminated?
			for key in list(self.active_nodes.keys()):
				if self.active_nodes[key].primal_value > self.curr_best_int_primal:
					prin(deleting=key)
					prin(deleted_node_primal=self.active_nodes[key].primal_value)
					del self.active_nodes[key]
	

	def step(self, action):
		"""
		action: an integer value and a string. The integer value is the x_i index and the string is the node key.  
			The x_i index is relative to p.
			The integer solution search subroutine is indicated by the x_i index == -1.
		"""
# TODO: do we need to change the seed if we are searching twice in the same node?
	
		if int(action[0]) < 0:
			### Search
			self.search_counter += 1 
			node = self.active_nodes[action[1]]
			search_support, search_betas = \
				get_search_solution(node=node, p=self.p, l0=self.l0, l2=self.l2, y=self.y)
			self.eliminate_after_search(search_betas, search_support)

		else:
			### Branch
			branching_x_i = int(action[0])
			branching_node = self.active_nodes[action[1]]

			new_zlb = branching_node.zlb.copy()
			new_zlb.append(branching_x_i)
		
			new_zub = branching_node.zub.copy()
			new_zub.append(branching_x_i)
			self.node_counter += 1
			node_name_1 = f'node_{self.node_counter}'

			self.active_nodes[node_name_1] = \
				Node(parent=branching_node, zlb=new_zlb, zub=branching_node.zub)
			self.node_counter +=1
			node_name_2 = f'node_{self.node_counter}'
			self.active_nodes[node_name_2] = \
				Node(parent=branching_node, zlb=branching_node.zlb, zub=new_zub)

			del self.active_nodes[action[1]] 

			### Solve relaxations in new nodes
			# TODO: look into the following (from 'tree.solve') ...
			# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"
			self.active_nodes[node_name_1].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
			self.active_nodes[node_name_2].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
			
			### Check to see if any nodes can be eliminated
			self.eliminate_nodes(node_key=node_name_1) # Note: this might eliminate the 2nd new node
			if self.active_nodes.get(node_name_2, 0) != 0:
				self.eliminate_nodes(node_key=node_name_2)

		### Check if all nodes have been eliminated
		if len(self.active_nodes) == 0:
			# Gather return values
			done = True
			newline = '\n'
			observation = f'primal value: {self.curr_best_int_primal}{newline} \
					beta: {self.curr_best_int_beta}{newline}support: {self.curr_best_int_support}'
			reward = -self.node_counter - self.alpha * self.search_counter
			info = f'node count: {self.node_counter}{newline} search count: {self.search_counter}'
			return(observation, reward, done, info)

		### Stats
		global_stats = np.array([self.l0, self.l2, self.p])		
		active_x_i = []
		for node_key in self.active_nodes:
			active_x_i = active_x_i + \
				[k for k in range(self.p) if k not in self.active_nodes[node_key].zlb \
				and k not in self.active_nodes[node_key].zub and k not in active_x_i]
			if len(active_x_i) == self.p:
				break
		
		static_stats = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)
		branch_stats, search_stats, branch_keys, search_keys = \
			get_action_specific_stats(self.active_nodes, self.p, self.cov, self.x, self.y)
		
		# Capping the number of branching actions passed to the branching model at p**2
		n_branch_actions = branch_stats.shape[0]
		if n_branch_actions > self.p**2:
			random_ind = np.random.choice(n_branch_actions, size=self.p**2, replace=False)
			branch_stats = branch_stats[random_ind, :]
			n_branch_actions = self.p**2 # Do we need this line?
		
		# Capping the number of searching actions passed to the searching model at p**2
		n_search_actions = search_stats.shape[0]
		if n_search_actions > self.p**2:
			random_ind = np.random.choice(n_search_actions, size=self.p**2, replace=False)
			search_stats = search_stats[random_ind, :]
			n_search_actions = self.p**2 # Do we need this line?

		# Get q_hats by applying models
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, n_branch_actions, static_stats)
		search_q_hats = get_q_hats(self.search_model_name, search_stats, n_search_actions, static_stats)

		### Gather return values
		action_q_hats = np.concatenate((branch_q_hats, search_q_hats))
		action_keys  =  np.concatenate((branch_keys, search_keys))
	
		done = False
		observation = [action_q_hats, action_keys] 
		reward = 0
		# TODO: formating with newline is not working
		newline = '\n'
		info = f'node count: {self.node_counter}{newline} search count: {self.search_counter}'
		return(observation, reward, done, info)












