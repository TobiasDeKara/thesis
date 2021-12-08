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

# TODO: change q_array in get record to total_cost - self.cost_so_far
class action_taken:
	def __init__(self, alpha, prev_action,  branch_or_search, static_stats, specific_stats, q_hat, step_number):
		self.alpha = alpha
		self.prev_action = prev_action
		self.branch_or_search = branch_or_search
		self.static_stats = static_stats
		self.specific_stats = specific_stats
		self.q_hat = q_hat
		self.step_number = step_number
		self.cost_of_action = 2*(branch_or_search == 'branch') + self.alpha*(branch_or_search == 'search')
		if prev_action is None:
			self.cost_so_far = self.cost_of_action
		else:
			self.cost_so_far = prev_action.cost_so_far + self.cost_of_action

	def get_record(self, total_cost):
# TODO: leaving off here
		q_array = np.array([self.q_hat, -self.cost_so_far], dtype=float)
		record = np.concatenate([self.static_stats, self.specific_stats, q_array])
		return(record)
class rl_env:
	def __init__(self, l0=10**-4, l2=1, p=10**3, m=5, alpha=1, greedy_epsilon=0.3, \
			branch_model_name='branch_model_in58_lay2_0', \
			search_model_name='search_model_in49_lay2_0'):
		""" greedy_epsilon is the probability of choosing random exploration, for testing set to zero."""
		self.l0 = l0
		self.l2 = l2
		self.p = p
		self.branch_model_name = branch_model_name
		self.search_model_name = search_model_name
		self.greedy_epsilon = greedy_epsilon

		# For mini data (p=5)
		if p==5:
			self.x_file_list = subprocess.run( \
				f"cd synthetic_data; ls x*_pmini_* -1U", \
				capture_output=True, text=True, shell=True).stdout.splitlines()
		# For all other data (p in {10**3, 10**4, 10**5, 10**6})
		else:
			self.x_file_list = subprocess.run( \
				f"cd synthetic_data; ls x*_p{int(np.log10(self.p))}_* -1U", \
				capture_output=True, text=True, shell=True).stdout.splitlines()

		self.int_tol = 10**-4
		# m=5 works well for our synthetic data, but will need to adjusted for other data sets.
		self.m = m	
		self.alpha = alpha

		# All of the following attributes are reset in 'reset()' below.
		self.x_file_name = None
		self.active_nodes = None
		self.node_counter = 0
		self.search_counter = 0
		self.step_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None
		self.current_action = None
		self.state = None

	def reset(self, x_file_name=None):
		"""
		Creates the initial observation with a new x and y.
		""" 
		self.active_nodes = dict()
		self.node_counter = 0
		self.search_counter = 0
		self.step_counter = 0
		self.x = None
		self.y = None
		self.b = None
		self.cov = None
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None
		self.state = dict() 
		# self.state holds: static_stats, branch_option_stats, branch_option_keys, branch_q_hat,
		# search_option_stats, search_option_key, search_q_hat

		if x_file_name is None:
			# Randomly select data from the test bed, with the given number of variables, 'self.p'
			x_file_name = np.random.choice(self.x_file_list)

		self.x_file_name = x_file_name
	
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
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		self.state['static_stats'] = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)

		### Begin epsilon greedy policy
		random_number = np.random.random()
		if random_number < self.greedy_epsilon:
			# Choose random branch and search actions, and get stats
			branch_stats, search_stats, branch_keys, search_keys = \
				get_random_action_stats(self.active_nodes, self.p, self.cov, self.x, self.y)
		else:
			# Get stats for all available actions
			branch_stats, search_stats, branch_keys, search_keys = \
				get_all_action_stats(self.active_nodes, self.p, self.cov, self.x, self.y)
		
		# Get q_hats by applying models
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, self.state['static_stats'])
		search_q_hats = get_q_hats(self.search_model_name, search_stats, self.state['static_stats'])

		# Record stats, keys and q_hats for the branch and search actions passed to the agent
		self.record_action_stats(branch_stats, branch_keys, search_stats, search_keys, branch_q_hats, search_q_hats)

		# Gather return values
		observation = np.concatenate([self.state['static_stats'], self.state['branch_option_stats'], \
			self.state['search_option_stats']])

		return(observation)


	def get_info(self):
		return(f'node count: {self.node_counter} \nsearch count: {self.search_counter} \nstep count: {self.step_counter} \nx_file_name: {self.x_file_name}')

	def eliminate_nodes(self, node_key):
		node = self.active_nodes[node_key]
		
		# Can the node be eliminated?
		if self.curr_best_int_primal < node.primal_value:
			del self.active_nodes[node_key]

		# Is the  new node an integer solution?
		# If so, then it is the new best integer solution, because otherwise it would have been
		# eliminated in the step above. 
		elif int_sol(node, p=self.p, int_tol=self.int_tol, m=self.m):
			self.curr_best_int_primal = node.primal_value
			self.curr_best_int_beta = node.primal_beta.copy()
			self.curr_best_int_support = node.support.copy()
			del self.active_nodes[node_key]

			# Can other nodes be eliminated?
			for other_key in list(self.active_nodes.keys()):
				if self.active_nodes[other_key].primal_value > self.curr_best_int_primal:
					del self.active_nodes[other_key]

	def eliminate_after_search(self, search_betas, search_support):
		if search_support.shape[0] == 1:
			residuals = self.y - np.dot(self.x[:,search_support],  search_betas)
		else:
			residuals = self.y - np.matmul(self.x[:,search_support], search_betas)

		rss = np.dot(residuals, residuals)

		search_sol_primal = rss/2 + self.l0*search_support.shape[0] + self.l2*np.dot(search_betas, search_betas)

		if search_sol_primal < self.curr_best_int_primal:
			# Update current best integer solution
			self.curr_best_int_primal = search_sol_primal.copy()
			self.curr_best_int_beta = search_betas.copy()
			self.curr_best_int_support = search_support.copy()

			# Can other nodes be eliminated?
			for key in list(self.active_nodes.keys()):
				if self.active_nodes[key].primal_value > self.curr_best_int_primal:
					del self.active_nodes[key]
	
	def record_action_stats(self, branch_stats, branch_keys, search_stats, search_keys, branch_q_hats, search_q_hats):
		""" Selects the branching option and searching option with highest q_hats, and 
			attaches to the rl_env the stats, keys, and q_hats for the chosen actions and """
		branch_ind = np.argmax(branch_q_hats)
		self.state['branch_option_stats'] = branch_stats[branch_ind, :]
		self.state['branch_option_keys'] = branch_keys[branch_ind, :]
		self.state['branch_q_hat'] = branch_q_hats[branch_ind]
		search_ind = np.argmax(search_q_hats)
		self.state['search_option_stats'] = search_stats[search_ind, :]
		self.state['search_option_key'] = search_keys[search_ind]
		self.state['search_q_hat'] = search_q_hats[search_ind]

	def step(self, action):
		"""
		action: 0 or 1, 0 indicates that the branching option has been chosen, 1 indciates the searching option
		"""
		self.step_counter += 1

		### Update self.current_action
		prev_action = self.current_action if self.step_counter > 1 else None
		if action == 0:
			# branching option chosen
			branch_or_search = 'branch'
			specific_stats = self.state['branch_option_stats']
			q_hat = self.state['branch_q_hat']
		else:
			# searching option chosen
			branch_or_search = 'search'
			specific_stats = self.state['search_option_stats']
			q_hat = self.state['search_q_hat']

		self.current_action = action_taken(self.alpha, prev_action, branch_or_search, self.state['static_stats'], \
				specific_stats, q_hat, self.step_counter)
		
		### Implement action
		if branch_or_search == 'search':
			# Search
			self.search_counter += 1
			search_node_key = self.state['search_option_key']
			search_node = self.active_nodes[search_node_key]
			search_support, search_betas = \
				get_search_solution(node=search_node, p=self.p, l0=self.l0, l2=self.l2, y=self.y)
			self.eliminate_after_search(search_betas, search_support)

		else:
			# Branch
			branch_x_i = int(self.state['branch_option_keys'][0])
			branch_node_key = self.state['branch_option_keys'][1]
			branch_node = self.active_nodes[branch_node_key]

			new_zlb = branch_node.zlb.copy()
			new_zlb.append(branch_x_i)
			new_zub = branch_node.zub.copy()
			new_zub.append(branch_x_i)
			self.node_counter += 1
			node_name_1 = f'node_{self.node_counter}'
			self.active_nodes[node_name_1] = \
				Node(parent=branch_node, zlb=new_zlb, zub=branch_node.zub)
			self.node_counter +=1
			node_name_2 = f'node_{self.node_counter}'
			self.active_nodes[node_name_2] = \
				Node(parent=branch_node, zlb=branch_node.zlb, zub=new_zub)

			del self.active_nodes[branch_node_key] 

			### Solve relaxations in new nodes
			# TODO: look into the following (from 'tree.solve') ...
			# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"
			self.active_nodes[node_name_1].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
			self.active_nodes[node_name_2].lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
			
			### Check to see if any nodes can be eliminated
			self.eliminate_nodes(node_key=node_name_1) # Note: this might eliminate the 2nd new node
			if self.active_nodes.get(node_name_2, 0) != 0:
				self.eliminate_nodes(node_key=node_name_2)

		### If we're done ...
		if len(self.active_nodes) == 0:
			# Get record of most recent action taken
			branch_action_records, search_action_records = [], []
			action = self.current_action
			total_cost = action.cost_so_far.copy()
			if action.branch_or_search == 'branch':
				branch_action_records.append(action.get_record(total_cost).copy())
			else:
				search_action_records.append(action.get_record(total_cost).copy())
			# Get records of previous actions taken
			while action.prev_action is not None:
				action = action.prev_action
				if action.branch_or_search == 'branch':
					branch_action_records.append(action.get_record(total_cost).copy())
				else:
					search_action_records.append(action.get_record(total_cost).copy())
			# Save records
			if branch_action_records:
				branch_action_records = np.vstack(branch_action_records)
				branch_record_dim = branch_action_records.shape[1]
				branch_file_name = f'branch_action_records_dim{branch_record_dim}_{self.x_file_name}'
				np.save(os.path.join('action_records', branch_file_name), branch_action_records)

			if search_action_records:
				search_action_records = np.vstack(search_action_records)
				search_record_dim = search_action_records.shape[1]
				search_file_name = f'search_action_records_dim{search_record_dim}_{self.x_file_name}'
				np.save(os.path.join('action_records', search_file_name), search_action_records)

			# Gather return values
			done = True
			observation = f'primal value: {self.curr_best_int_primal} \nbeta: {self.curr_best_int_beta} \nsupport: {self.curr_best_int_support}'
			reward = -self.current_action.cost_so_far
			info = self.get_info()
	
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
		
		self.state['static_stats'] = get_static_stats(self.cov, self.x, self.y, self.active_nodes, active_x_i, global_stats)

		### Begin epsilon greedy policy
		random_number = np.random.random()
		if random_number < self.greedy_epsilon:
			# Choose random branch and search actions, and get stats
			branch_stats, search_stats, branch_keys, search_keys = \
				get_random_action_stats(self.active_nodes, self.p, self.cov, self.x, self.y)
		else:
			# Get stats for all available actions
			branch_stats, search_stats, branch_keys, search_keys = \
				get_all_action_stats(self.active_nodes, self.p, self.cov, self.x, self.y)
		
			# Cap the number of branch and search actions passed to the q models at p**2
			if branch_stats.shape[0] > self.p**2:
				random_ind = np.random.choice(branch_stats.shape[0], size=self.p**2, replace=False)
				branch_stats = branch_stats[random_ind, :]
			if search_stats.shape[0] > self.p**2:
				random_ind = np.random.choice(search_stats.shape[0], size=self.p**2, replace=False)
				search_stats = search_stats[random_ind, :]

		# Get q_hats by applying models
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, self.state['static_stats'])
		search_q_hats = get_q_hats(self.search_model_name, search_stats, self.state['static_stats'])

		# Record stats, keys and q_hats for the branch and search actions passed to the agent
		self.record_action_stats(branch_stats, branch_keys, search_stats, search_keys, branch_q_hats, search_q_hats)

		# Gather return values
		observation = np.concatenate([self.state['static_stats'], self.state['branch_option_stats'], \
			self.state['search_option_stats']])
		done = False
		reward = 0
		info = self.get_info()
		return(observation, reward, done, info)












