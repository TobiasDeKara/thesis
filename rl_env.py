# This script implements a reinforcement learning environment,
# in the format of OpenAI's package 'gym'.

# Much of the code below is taken directly from 'l0bnb', 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import numpy as np
from copy import copy
from l0bnb.node import Node
from random import choice
from functions import *
from stat_functions import *
import math
from action_taken import action_taken
from operator import attrgetter
import re

# Note on optimality gap:
# optimality gap = (upper bound - lower bound) / lower bound
# The upper bound is defined as the minimum objective value of all known
# integer solutions, and below we use the variable 'curr_best_int_primal'.
# For each new node that is created by branching we use 'lower_solve' to find the 
# primal solution to the relaxation.    We then use 'upper_solve' which calculates
# an integer solution by taking the support of the relaxation solution and setting
# all z_i to 1.    Thus, we have one known integer solution per node in addition 
# to any integer solutions discovered using the search sub-routine.    The minimum
# of these is the upper bound.    When branching, the integer solution returned 
# by 'upper_solve' for each daughter node can be greater than, equal to, or less
# than the integer solution for the parent node.    So after every branch and every
# search we have to check for changes to the upper bound (again, referred to as
# 'curr_best_int_primal').    
# The lower bound is defined as the minimum primal objective value of relaxations
# over all active nodes.    Searching does not affect the primal values(and elimination 
# after searching cannot remove the node that has the minimal relaxation primal value).    
# When branching, the primal values of the daughter nodes are always greater than
# or equal to the primal value of the parent node.    Therefore, if we branch 
# on a node that does not have the current minimum primal, then the lower
# bound will not change.    We only have to check for changes to the lower bound
# when we branch the node that does have the current minimum primal.


class rl_env(gym.Env):
	def __init__(self, L0=10**-4, L2=1, p=10**3, m=5, greedy_epsilon=0.3, run_n=0, batch_n=0, \
	branch_model_name='branch_model_in62_lay6_drop_out_yes_rew_binary_reg_True_rate_1e-05_range',
	mp=False, test_train='train'):
		super(rl_env, self).__init__()
		""" Note: 'greedy_epsilon' is the probability of choosing random exploration, 
		for testing performance after training, set greedy_epsilon to zero."""
		self.mp = mp
		self.test_train = test_train
		self.L0 = L0
		self.log_L0 = -int(np.log10(L0))
		self.L2 = L2
		self.log_L2 = -int(np.log10(L2))
		self.p = int(p)
		self.log_p = int(np.log10(int(p)))
		self.branch_model_name = branch_model_name
		self.reward = re.search('(binary|numeric|sig_num)', branch_model_name)[1]
		self.greedy_epsilon = greedy_epsilon
		self.run_n = run_n
		self.batch_n = batch_n 
		# A 'batch' index is used for each sub-directory of 'thesis/synthetic_data/{p_sub_dir}' 
		# that has its own subset of the training data.  A 'run' index is used to indicated how many
		# times the q_model has been updated, and a run will involve using several batches of data.
		# Each batch of training data is assigned 3 
		# workers, one for each of 3 values of the L0 penalty.  
		# Each worker (or each combination of batch index and L0 penalty value) is given its own 
		# subdirectories for 1. passing parameters to the search subroutine, 2. collecting results
		# from the search subroutine, and 3. calling copies of the q_model.

		# For mini data (p=5)
		if self.p==5:
			self.p_sub_dir = 'pmini' 
		# For all other data (p in {10**3, 10**4, 10**5, 10**6})
		else:
               	    self.p_sub_dir = f'p{int(np.log10(self.p))}'

		data_dir = f'/gpfs/scratch/tdekara/synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}'
		self.x_file_list = [f for f in os.listdir(data_dir) if re.match('x', f)]

		# Make copies of q-models
		subprocess.run(f'cp -r models/{self.branch_model_name} model_copies/batch_{batch_n}/L0_{self.log_L0}_L2_{self.log_L2}', shell=True)

		self.int_tol = 10**-4
		# m=5 works well for our synthetic data, but will need to adjusted for other data sets.
		self.m = m

		# All of the following attributes are reset in 'reset()' below.
		self.x_file_name = None
		self.active_nodes = None
		self.step_counter = 0
		self.node_counter = 0
		self.x = None
		self.y = None
		self.cov = None
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None
		self.current_action = None
		self.state = None    
		# Note: self.state is the representation of the state passed to agent (see comment below).
		self.lower_bound = None    # The minimum primal value over all active nodes.    This is the best
		# case scenario, i.e. how good any integer solution yet to be found could be.
		self.lower_bound_node_key = None # The key of the node with the lowest primal value
		self.upper_bound_node_key = None 
		# Node with lowest upper bound, if the upper bound is not an integer primal solution, else None

		self.initial_optimality_gap = None
		self.optimality_gap = None
		self.record_batch_counter = 0

	def reset(self, x_file_name=None):
		"""
		Creates the initial observation with a new x and y.
		""" 
		self.active_nodes = dict()
		self.step_counter = 0
		self.node_counter = 1
		self.x = None
		self.y = None
		self.cov = None
		self.curr_best_int_primal = math.inf
		self.curr_best_int_beta = None
		self.curr_best_int_support = None
		self.state = dict() 
		# self.state holds: static_stats, branch_option_stats, branch_option_keys, and branch_q_hat
		self.record_batch_counter = 0

		if x_file_name is None:
                    if len(self.x_file_list) == 0:
                        data_dir = f'/gpfs/scratch/tdekara/synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}'
                        self.x_file_list = [f for f in os.listdir(data_dir) if re.match('x', f)]

                    rand_ind = np.random.choice(len(self.x_file_list))
                    x_file_name = self.x_file_list.pop(rand_ind)

		self.x_file_name = x_file_name
	
		y_file_name = x_file_name.replace('x', 'y')
		self.x = np.load(f'/gpfs/scratch/tdekara/synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}/{x_file_name}')
		self.y = np.load(f'/gpfs/scratch/tdekara/synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}/{y_file_name}')
		self.y = self.y.reshape(1000)
		
		### Initialize root node
		xi_norm =  np.linalg.norm(self.x, axis=0) ** 2
		root_node = Node(parent=None, zlb=[], zub=[], x=self.x, y=self.y, xi_norm=xi_norm)
		self.active_nodes['root_node'] = root_node
		active_x_i = list(range(self.p))
		# Note: 'lower_solve' returns the primal value and dual value, and it updates
		# root_node.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		root_node.lower_solve(self.L0, self.L2, m=5, solver='l1cd', rel_tol=1e-8, mio_gap=1e-2)
		self.lower_bound = root_node.primal_value
		self.lower_bound_node_key = 'root_node'
		self.upper_bound_node_key = 'root_node'
		root_node.upper_solve(self.L0, self.L2, m=5)
		self.update_curr_best_int_sol(root_node, 'upper')
		self.initial_optimality_gap = \
		    (self.curr_best_int_primal - self.lower_bound) / self.lower_bound
		self.optimality_gap = \
		    (self.curr_best_int_primal - self.lower_bound) / self.lower_bound

		### Use coordinate descent search
#		search_node = self.active_nodes['root_node']
#		search_support, search_betas = \
#			get_search_solution(node=search_node, p=self.p, log_L0=self.log_L0, \
#			log_L2=self.log_L2, y=self.y, batch_n=self.batch_n, log_p=self.log_p)
#
#		    # Find primal value of search solution	
#		if search_support.shape[0] == 1:
#			residuals = self.y - np.dot(self.x[:,search_support], search_betas)
#		else:
#			residuals = self.y - np.matmul(self.x[:,search_support], search_betas)
#		rss = np.dot(residuals, residuals)
#		search_sol_primal = rss/2 + self.L0*search_support.shape[0] + \
#		self.L2*np.dot(search_betas, search_betas)
#
#		# Check if new solution is best so far
#		if search_sol_primal < self.curr_best_int_primal:
#			# Update current best integer solution
#			self.curr_best_int_primal = search_sol_primal
#			self.curr_best_int_beta = search_betas.copy()
#			self.curr_best_int_support = search_support.copy()

		#### End of Search routine

		#### Gather stats to describe the state
		global_stats = \
			np.array([self.L0, self.L2, self.p, self.initial_optimality_gap], dtype=float)
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		self.state['static_stats'] = get_static_stats(self.cov, self.x, self.y, \
		    	self.active_nodes, active_x_i, global_stats)

		### Begin epsilon greedy policy
		random_number = np.random.random()
		if random_number < self.greedy_epsilon:
			# Choose random action, and get stats
			branch_stats, branch_keys = \
				get_random_action_stats(self.active_nodes, self.p, self.cov, \
				self.x, self.y, self.lower_bound_node_key, self.upper_bound_node_key)
		else:
			# Get stats for all available actions
			branch_stats, branch_keys = \
				get_all_action_stats(self.active_nodes, self.p, self.cov, \
				self.x, self.y, self.lower_bound_node_key, self.upper_bound_node_key)
		
		# Get q_hats by applying models
		branch_q_hats = \
			get_q_hats(self.branch_model_name, branch_stats, \
			self.state['static_stats'], self.batch_n, self.log_L0, self.log_L2)

		# Record stats and key
		self.attach_action_option_stats(branch_stats, branch_keys, branch_q_hats)

		# testing
		# return(self.get_info())
    	#### End of reset() #######

	def get_info(self):
		info = {'step_count': self.step_counter,
			'node_count': self.node_counter,
			'x_file_name': self.x_file_name,
			'curr_best_primal_value': self.curr_best_int_primal,
			'beta': self.curr_best_int_beta,
			'support': self.curr_best_int_support,
			'lower_bound': self.lower_bound,
			'init_opt_gap' : self.initial_optimality_gap,
			'opt_gap': self.optimality_gap}
		return(info)

	def attach_action_option_stats(self, branch_stats, branch_keys, branch_q_hats):
		""" Attaches the stats, keys, and q_hats of the selected action
		to the rl_env """
		# Select options to be passed to agent by taking arg max of q-hats
		# Note that when (random_number < greedy_epsilon) only one option is passed in,
		# and taking the arg max in that case is not strictly necessary.
		branch_ind = np.argmax(branch_q_hats) 
		self.state['branch_option_stats'] = branch_stats[branch_ind, :]
		self.state['branch_option_keys'] = branch_keys[branch_ind, :]
		self.state['branch_q_hat'] = branch_q_hats[branch_ind]
        
	def update_curr_best_int_sol(self, node, primal_or_upper):
	    """ Updates whenever we find an integer solution that is better than 
	    the current best integer solution.
	    primal_or_upper: 'primal' if the integer solution was discovered 
	    as the primal solution of a new node after branching, else 'upper'"""

	    if primal_or_upper == 'primal':
		    self.curr_best_int_primal = node.primal_value
		    self.curr_best_int_beta = node.primal_beta.copy()
		    self.curr_best_int_support = node.support.copy()
	    else:
		    self.curr_best_int_primal = node.upper_bound
		    self.curr_best_int_beta = node.upper_beta.copy()
		    self.curr_best_int_support = node.support.copy()

	def step(self):
		self.step_counter += 1
		
		### Implement branching action
		branch_x_i = int(self.state['branch_option_keys'][0])
		branch_node_key = self.state['branch_option_keys'][1]
		branch_node = self.active_nodes[branch_node_key]

		# Create daughter nodes
		new_zlb = branch_node.zlb.copy()
		new_zlb.append(branch_x_i)
		new_zub = branch_node.zub.copy()
		new_zub.append(branch_x_i)

		node_name_1 = f'node_{self.node_counter}'
		self.active_nodes[node_name_1] = \
			Node(parent=branch_node, zlb=new_zlb, zub=branch_node.zub, \
			x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
		self.node_counter += 1

		node_name_2 = f'node_{self.node_counter}'
		self.active_nodes[node_name_2] = \
			Node(parent=branch_node, zlb=branch_node.zlb, zub=new_zub, \
			x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
		del self.active_nodes[branch_node_key] 
		self.node_counter += 1

		# Solve relaxations in new nodes
		self.active_nodes[node_name_1].lower_solve(\
			self.L0, self.L2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		self.active_nodes[node_name_2].lower_solve(\
			self.L0, self.L2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		# Update current best integer solution (aka upper bound)
		upper_bound_updated = False
		# Check if node 1 relaxation solution is integer, and if so check if best so far
		if int_sol(self.active_nodes[node_name_1], p=self.p, int_tol=self.int_tol, m=self.m):
			if self.active_nodes[node_name_1].primal_value < self.curr_best_int_primal:
				self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'primal')
				self.upper_bound_node_key = None
				upper_bound_updated = True
			del self.active_nodes[node_name_1]
		else:    # If not int., find node 1 upper bound, and check if best so far
			n_1_ub = self.active_nodes[node_name_1].upper_solve(self.L0, self.L2, m=5)
			if n_1_ub < self.curr_best_int_primal:
				self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'upper')
				self.upper_bound_node_key = node_name_1
				upper_bound_updated = True
		# repeat for node 2
		if int_sol(self.active_nodes[node_name_2], p=self.p, int_tol=self.int_tol, m=self.m):
			if self.active_nodes[node_name_2].primal_value < self.curr_best_int_primal:
				self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'primal')
				self.upper_bound_node_key = None
				upper_bound_updated = True
			del self.active_nodes[node_name_2]
		else:
			n_2_ub = self.active_nodes[node_name_2].upper_solve(self.L0, self.L2, m=5)
			if n_2_ub < self.curr_best_int_primal:
				self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'upper')
				self.upper_bound_node_key = node_name_2
				upper_bound_updated = True

		# Check for eliminations
		if upper_bound_updated:
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
	
		# Update lower_bound and lower_bound_node_key
		if branch_node_key == self.lower_bound_node_key:
			if len(self.active_nodes) == 0:
				self.lower_bound = self.curr_best_int_primal
			else:
				self.lower_bound = \
					min(self.active_nodes.values(), \
					key=attrgetter('primal_value')).primal_value
				self.lower_bound_node_key = reverse_lookup(self.active_nodes, \
					min(self.active_nodes.values(), key=attrgetter('primal_value')))
		# else: lower_bound remains unchanged
		    
		    ###### End of Implementing Branch Action ####

		### Calculate change in optimality gap 
		prev_opt_gap = self.optimality_gap
		self.optimality_gap = (self.curr_best_int_primal - self.lower_bound) / self.lower_bound
		frac_change_in_opt_gap = (prev_opt_gap - self.optimality_gap) / self.initial_optimality_gap
		# The following two lines are for finite precision issues.  When the 
		# support of the primal solution to the relaxation of the root node is 
		# the same as the support of the optimal integer solution the initial 
		# optimality gap can be small relative to the precision of the calculations. 
		frac_change_in_opt_gap = max(frac_change_in_opt_gap, 0)
		frac_change_in_opt_gap = min(frac_change_in_opt_gap, 1)
		
		### Attach stats on action taken <==> Update self.current_action
		specific_stats = self.state['branch_option_stats']
		q_hat = self.state['branch_q_hat']
		    
		prev_action = self.current_action if self.step_counter > 1 else None
		
		self.current_action = action_taken(prev_action,  \
			    self.state['static_stats'], specific_stats, q_hat, self.step_counter, \
			    frac_change_in_opt_gap)

		### If we're done ...
		if len(self.active_nodes) == 0:
			### Gather and save records
			action = self.current_action
			total_n_nodes = self.node_counter
			total_n_steps = self.step_counter

			# Compare model support to true support => ep_res_record
			seed_support_array = \
				np.load(f'combined_seed_support_records/seed_support_rec_comb_p{self.log_p}.npy')
			seed = re.search('(?<=seed)[0-9]*', self.x_file_name)[0]
			seed = int(seed)
			true_support = \
				seed_support_array[np.where(seed_support_array[:,0]==seed),1:].reshape(-1)
			len_model_support = len(self.curr_best_int_support)
			sum_true_sup_in_mod_sup = 0
			for x_i in true_support:
				if x_i in self.curr_best_int_support:
					sum_true_sup_in_mod_sup += 1
			if true_support.shape[0] > 0:
				frac_true_sup_in_mod_sup = sum_true_sup_in_mod_sup / true_support.shape[0]
			else:
				frac_true_sup_in_mod_sup = 'NA'
			ep_res_record = \
				np.array([self.run_n, seed, self.p, self.L0, self.L2, \
					self.branch_model_name,\
					 total_n_nodes, len_model_support, frac_true_sup_in_mod_sup])
			data_info = re.sub('x_', '', self.x_file_name)
			data_info = re.sub('.npy', '', data_info)
			data_info = data_info + 'L0_' +  str(self.log_L0) + '_L2_' + str(self.log_L2)
		    
			if not self.mp:
				np.save(f'./ep_res_records/run_{self.run_n}/ep_res_rec_{self.reward}_{data_info}_{self.branch_model_name}', ep_res_record)

			# Get records of most recent action taken
			branch_action_records = []
			branch_model_records = []

			action_record = np.concatenate([action.static_stats, \
				action.specific_stats])
			action_record = np.append(action_record, action.frac_change_in_opt_gap)

			model_record = get_model_record(self.run_n, self.p, self.L0, self.L2, action)

			branch_action_records.append(action_record)
			branch_model_records.append(model_record)

			# Get records of other recent actions (up to 10 previous)
			for _ in range(total_n_steps % 10):
				if action.prev_action is not None:
					prev_action = action.prev_action
					del action
					action = prev_action
					action_record = np.concatenate([action.static_stats, \
					action.specific_stats])
					action_record = np.append(action_record, action.frac_change_in_opt_gap)

					model_record = get_model_record(self.run_n, self.p, \
								self.L0, self.L2, action)

					branch_action_records.append(action_record)
					branch_model_records.append(model_record)
	
			data_info = re.sub('x_', '', self.x_file_name)
			data_info = re.sub('.npy', '', data_info)
			data_info = data_info + 'L0_' +  str(self.log_L0) + '_L2_' + str(self.log_L2)
	
			# Save action records
			if branch_action_records:
				branch_action_records = np.vstack(branch_action_records)
				branch_record_dim = branch_action_records.shape[1]
				file_name = \
					f'branch_action_rec_dim{branch_record_dim}_{data_info}_{self.record_batch_counter}'
				if not self.mp and self.test_train == 'train':
					np.save(f'action_records/run_{self.run_n}/{file_name}', branch_action_records)
				if self.mp and self.test_train == 'train':
					np.save(f'/gpfs/home/tdekara/thesis/replay_buffer/{file_name}', branch_action_records)
				del branch_action_records
	
				branch_model_records = np.vstack(branch_model_records)
				branch_record_dim = branch_model_records.shape[1]
				model_data_info =  f'{data_info}_{self.branch_model_name}'
				file_name = f'branch_model_rec_dim{branch_record_dim}_{model_data_info}_{self.record_batch_counter}'
				if not self.mp:
					np.save(f'model_records/run_{self.run_n}/{file_name}', branch_model_records)
				del branch_model_records
				#### End Gather and Save Records #######

			# info = self.get_info()
			done = True
			return(done)
			# return(done, info)
			### End of "If we're done" #########
		    
		    
		### If we're NOT done . . . 
		
		# Write to file action and model stats from last 10 actions
		if self.step_counter % 10 == 0: # and self.test_train == 'train':
			# Get records of most recent action taken
			action = self.current_action
			branch_action_records = []
			branch_model_records = []

			action_record = np.concatenate([action.static_stats, \
				action.specific_stats])
			action_record = np.append(action_record, action.frac_change_in_opt_gap)

			model_record = get_model_record(self.run_n, self.p, self.L0, self.L2, action)

			branch_action_records.append(action_record)
			branch_model_records.append(model_record)

			# Get action records of previous actions taken
			for _ in range(9):
				prev_action = action.prev_action
				del action
				action = prev_action
				action_record = np.concatenate([action.static_stats, \
					action.specific_stats])
				action_record = np.append(action_record, action.frac_change_in_opt_gap)

				model_record = get_model_record(self.run_n, self.p, self.L0, \
							self.L2,  action)

				branch_action_records.append(action_record)
				branch_model_records.append(model_record)

			data_info = re.sub('x_', '', self.x_file_name)
			data_info = re.sub('.npy', '', data_info)
			data_info = data_info + 'L0_' +  str(self.log_L0) + '_L2_' + str(self.log_L2)

			# Save action records
			if branch_action_records:
				branch_action_records = np.vstack(branch_action_records)
				branch_record_dim = branch_action_records.shape[1]
				file_name = \
					f'branch_action_rec_dim{branch_record_dim}_{data_info}_{self.branch_model_name}_{self.record_batch_counter}'
				if not self.mp and self.test_train == 'train':
					np.save(f'action_records/run_{self.run_n}/{file_name}', branch_action_records)
				if self.mp and self.test_train == 'train':
					np.save(f'/gpfs/home/tdekara/thesis/replay_buffer/{file_name}', branch_action_records)
				del branch_action_records

				branch_model_records = np.vstack(branch_model_records)
				branch_record_dim = branch_model_records.shape[1]
				model_data_info =  f'{data_info}_{self.branch_model_name}'
				file_name = f'branch_model_rec_dim{branch_record_dim}_{model_data_info}_{self.record_batch_counter}'
				if not self.mp:
					np.save(f'model_records/run_{self.run_n}/{file_name}', branch_model_records)
				del branch_model_records

			self.record_batch_counter += 1


		### Gather Stats
		global_stats = np.array([self.L0, self.L2, self.p, self.initial_optimality_gap])		
		active_x_i = []
		for node_key in self.active_nodes:
		    active_x_i = active_x_i + \
		    	[k for k in range(self.p) if k not in self.active_nodes[node_key].zlb \
		    	and k not in self.active_nodes[node_key].zub and k not in active_x_i]
		    if len(active_x_i) == self.p:
		    	break
		
		self.state['static_stats'] = get_static_stats(self.cov, self.x, self.y, \
		    	self.active_nodes, active_x_i, global_stats)

		### Begin epsilon greedy policy
		random_number = np.random.random()
		if random_number < self.greedy_epsilon:
			# Choose random action, and get stats
			branch_stats, branch_keys = \
				get_random_action_stats(self.active_nodes, self.p, self.cov, \
				self.x, self.y, self.lower_bound_node_key, self.upper_bound_node_key)
		else:
			# Get stats for all available actions (up to p**2)
			branch_stats, branch_keys = \
				get_all_action_stats(self.active_nodes, self.p, self.cov, \
				self.x, self.y, self.lower_bound_node_key, self.upper_bound_node_key)

		# Get q_hats by applying models
		# (to possilby 1 action option, all action options, or a capped # of action options)
		branch_q_hats = \
			get_q_hats(self.branch_model_name, branch_stats, \
			self.state['static_stats'], self.batch_n, self.log_L0, self.log_L2)

		# Record stats, keys and q_hats for the selected branch option
		self.attach_action_option_stats(branch_stats, branch_keys, branch_q_hats)

		# Gather return values
		done = False
		return(done)
		# info = self.get_info()
		# return(done, info)
