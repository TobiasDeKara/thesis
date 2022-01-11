# This script implements a reinforcement learning environment,
# in the format of OpenAI's package 'gym'.

# Much of the code below is taken directly from 'l0bnb', 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import gym
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
from action_taken import action_taken


# TODO: add change in optimality gap as reward
# optimality gap = (UB - LB) / UB
# initial LB = primal solution to initial relaxation
# initial UB = upper bound based on  solution of initial relaxation. It is the integer solution
# found by doing least squares regression using the support of the primal solution and setting
# all z_i of the support to 1.
# LB updates as min over active nodes of primal solutions, sometimes the prior min
# of primal solutions is no longer active (i.e. if it was the primal solution to 
# the relaxation of the node that was just branched) so, if branching the node with 
# current best primal then we have to compare across all nodes, otherwise we can just
# compare to the current best

# UB updates to min of searched integer solutions and upper bound integer solutions of active nodes
# but we can just track the one current best integer solutions because those
# don't go stale, i.e. even when a node is no longer active, the upper bound of that node is still in effect.
 
# TODO: record stats on model and agent performance


# class inherit from node, so that we can add an attribute for 'searched'
class rl_node(Node):
	def __init__(self, parent, zlb: list, zub: list, x, y, xi_norm):
		super().__init__(parent, zlb, zub, x=x, y=y, xi_norm=xi_norm)
		self.searched = 0


class rl_env(gym.Env):
	def __init__(self, l0=10**-4, l2=1, p=10**3, m=5, alpha=1, greedy_epsilon=0.3, \
			branch_model_name='branch_model_in60_lay2_0', \
			search_model_name='search_model_in51_lay2_0'):

		""" Note: 'greedy_epsilon' is the probability of choosing random exploration, 
		for testing performance after training, set greedy_epsilon to zero."""

		self.action_space = gym.spaces.Discrete(2)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(65,))
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
		self.state = None  # This is the representation of the state passed to agent (see comment below).
		self.lower_bound = None  # The minimum primal value over all active nodes.  This is the best
		# case scenario, i.e. how good any integer solution yet to be found could be.
		self.lower_bound_node_key = None # The key of the node with the lowest primal value

		# Note: If the lower_bound_node is selected for branching then the new lower_bound is found
		# as the min of primal values over all active nodes.  If any other node is selected for branching,
		# then the lower_bound is unchanged. This is because branching removes space from the feasible region and 
		# therefore can only increase or not change the primal value of the branched node, so if a node
		# was already not node with minimal primal value, neither daughter node can be either.
		self.optimality_gap = None

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
		t.root = rl_node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		self.active_nodes['root_node'] = t.root
		active_x_i = list(range(self.p))
		# Note: 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		t.root.lower_solve(self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		self.lower_bound = t.root.primal_value
		t.root.upper_solve(self.l0, self.l2, m=5)
		self.update_curr_best_int_sol(t.root, 'upper')
		self.optimality_gap = \
			(self.curr_best_int_primal - self.lower_bound) / abs(self.curr_best_int_primal)

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

	def update_curr_best_int_sol(self, node, primal_or_upper):
		if primal_or_upper == 'primal':
			self.curr_best_int_primal = node.primal_value
			self.curr_best_int_beta = node.primal_beta.copy()
			self.curr_best_int_support = node.support.copy()
		else:
			self.curr_best_int_primal = node.upper_bound
			self.curr_best_int_beta = node.upper_beta.copy()
			self.curr_best_int_support = node.support.copy()

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
			search_node.searched = 1
			search_support, search_betas = \
				get_search_solution(node=search_node, p=self.p, l0=self.l0, l2=self.l2, y=self.y)

			# Find primal value of search solution	
			if search_support.shape[0] == 1:
				residuals = self.y - np.dot(self.x[:,search_support],  search_betas)
			else:
				residuals = self.y - np.matmul(self.x[:,search_support], search_betas)
			rss = np.dot(residuals, residuals)
			# prin(rss=rss, search_support_shape=search_support.shape[0], search_betas=search_betas)
			search_sol_primal = rss/2 + self.l0*search_support.shape[0] + self.l2*np.dot(search_betas, search_betas)

			# Check for update to best integer solution
			if search_sol_primal < self.curr_best_int_primal:
				# Update current best integer solution
				self.curr_best_int_primal = search_sol_primal.copy()
				self.curr_best_int_beta = search_betas.copy()
				self.curr_best_int_support = search_support.copy()
				# Check all nodes for elimination
				for node_key in list(self.active_nodes.keys()):
					if self.active_nodes[node_key].primal_value > self.curr_best_int_primal:
						del self.active_nodes[node_key]
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
				rl_node(parent=branch_node, zlb=new_zlb, zub=branch_node.zub, \
				x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
			self.node_counter +=1
			node_name_2 = f'node_{self.node_counter}'
			self.active_nodes[node_name_2] = \
				rl_node(parent=branch_node, zlb=branch_node.zlb, zub=new_zub, \
				x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
			del self.active_nodes[branch_node_key] 

			# TODO: look into the following (from 'tree.solve') ...
			# "upper_bound, upper_beta, support = self._warm_start(warm_start, verbose, l0, l2, m)"

			### Solve relaxations in new nodes
			self.active_nodes[node_name_1].lower_solve(\
				self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
			self.active_nodes[node_name_2].lower_solve(\
				self.l0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

			### Update current best int solution
			updated = False
			# node 1
			if int_sol(self.active_nodes[node_name_1], p=self.p, int_tol=self.int_tol, m=self.m):
				if self.active_nodes[node_name_1].primal_value < self.curr_best_int_primal:
					self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'primal')
					updated = True
				del self.active_nodes[node_name_1]
			else:
				n_1_ub = self.active_nodes[node_name_1].upper_solve(self.l0, self.l2, m=5)
				if n_1_ub < self.curr_best_int_primal:
					self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'upper')
					updated = True
			# repeat for node 2
			if int_sol(self.active_nodes[node_name_2], p=self.p, int_tol=self.int_tol, m=self.m):
				if self.active_nodes[node_name_2].primal_value < self.curr_best_int_primal:
					self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'primal')
					updated = True
				del self.active_nodes[node_name_2]
			else:
				n_2_ub = self.active_nodes[node_name_2].upper_solve(self.l0, self.l2, m=5)
				if n_2_ub < self.curr_best_int_primal:
					self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'upper')
					updated = True

			# Check for eliminations
			if updated:
				# check all nodes for elimination
				for node_key in list(self.active_nodes.keys()):
					if self.active_nodes[node_key].primal_value > self.curr_best_int_primal:
						del self.active_nodes[node_key]
			else:
				# Check n_1 and n_2 for elimination
				for node_key in (node_name_1, node_name_2):
					if node_key in self.active_nodes:
						if self.active_nodes[node_key].primal_value > self.curr_best_int_primal:
							del self.active_nodes[node_key]
	
			# TODO:  pick up here, looking at getattr(), for below
			
			### Update lower_bound and lower_bound_node_key
#			if branch_node_key == lower_bound_node_key:
				# TODO: take minimum of primal values over active nodes
			# else: lower_bound remains unchanged
			

		### If we're done ...
		if len(self.active_nodes) == 0:
			# Get record of most recent action taken
			branch_action_records, search_action_records = [], []
			action = self.current_action
			total_cost = action.cost_so_far
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












