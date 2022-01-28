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
from l0bnb import BNBTree
from random import choice
from functions import *
from stat_functions import *
import math
from action_taken import action_taken
from operator import attrgetter
import re
import pickle

# TODO: add more sub_directories for records for different size p

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


# class inherit from node, so that we can add an attribute for 'searched'
class rl_node(Node):
	def __init__(self, parent, zlb: list, zub: list, x, y, xi_norm):
		super().__init__(parent, zlb, zub, x=x, y=y, xi_norm=xi_norm)
		self.searched = 0


class rl_env(gym.Env):
	def __init__(self, L0=10**-4, l2=1, p=10**3, m=5, greedy_epsilon=0.3, run_n=0, batch_n=0, \
	branch_model_name='branch_model_in60_lay2', \
	search_model_name='search_model_in51_lay2'):
		super(rl_env, self).__init__()
		""" Note: 'greedy_epsilon' is the probability of choosing random exploration, 
		for testing performance after training, set greedy_epsilon to zero."""

		self.action_space = gym.spaces.Discrete(2)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(65,))
		self.L0 = L0
		self.l2 = l2
		self.p = p
		self.branch_model_name = branch_model_name
		self.search_model_name = search_model_name
		self.greedy_epsilon = greedy_epsilon
		self.run_n = run_n
		self.batch_n = batch_n 
		# A 'batch' index is used for each sub-directory of 'thesis/synthetic_data/{p_sub_dir}' 
		# that has its own subset of the training data.  A 'run' index is used to indicated how many
		# times the q_models have been updated, and a run will involve using several batches of data.
		# When training using a vectorized environment, each batch of training data is assigned 3 
		# workers, one for each of 3 values of the L0 penalty.  
		# Each worker (or each combination of batch index and L0 penalty value) is given its own 
		# subdirectories for 1. passing parameters to the search subroutine, 2. collecting results
		# from the search subroutine, and 3. calling a copies of the q_models.

		# For mini data (p=5)
		if self.p==5:
			self.p_sub_dir = 'mini' 
		# For all other data (p in {10**3, 10**4, 10**5, 10**6})
		else:
               	    self.p_sub_dir = f'p{int(np.log10(self.p))}'

		self.x_file_list = subprocess.run( \
	    	f"cd synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}; ls x* -1U", \
	    	capture_output=True, text=True, shell=True).stdout.splitlines()

		self.int_tol = 10**-4
		# m=5 works well for our synthetic data, but will need to adjusted for other data sets.
		self.m = m

		# All of the following attributes are reset in 'reset()' below.
		self.x_file_name = None
		self.active_nodes = None
		self.node_counter = 0
		self.branch_counter = 0
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
		# Note: self.state is the representation of the state passed to agent (see comment below).
		self.lower_bound = None    # The minimum primal value over all active nodes.    This is the best
		# case scenario, i.e. how good any integer solution yet to be found could be.
		self.lower_bound_node_key = None # The key of the node with the lowest primal value
		self.optimality_gap = None

	def reset(self, x_file_name=None):
		"""
		Creates the initial observation with a new x and y.
		""" 
		self.active_nodes = dict()
		self.node_counter = 0
		self.branch_counter = 0
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
                    if len(self.x_file_list) == 0:
                        self.x_file_list = subprocess.run( \
                        f"cd synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}; ls x* -1U", \
                        capture_output=True, text=True, shell=True).stdout.splitlines()
                    x_file_name = self.x_file_list.pop(0)

		self.x_file_name = x_file_name
	
		y_file_name = x_file_name.replace('x', 'y')
		b_file_name = x_file_name.replace('x', 'b')
		self.x = np.load(f'synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}/{x_file_name}')
		self.y = np.load(f'synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}/{y_file_name}')
		self.y = self.y.reshape(1000)
		self.b = np.load(f'synthetic_data/{self.p_sub_dir}/batch_{self.batch_n}/{b_file_name}')
		global_stats = np.array([self.L0, self.l2, self.p], dtype=float)
		
		### Initialize a 'BNBTree' object (as defined in 'l0bnb'), and initialize its root node
		t = BNBTree(self.x, self.y) # TODO: Do I need this?
		# Is "xi_norm =  np.linalg.norm(x, axis=0) ** 2" all we need?
		t.root = rl_node(parent=None, zlb=[], zub=[], x=t.x, y=t.y, xi_norm=t.xi_norm)
		self.active_nodes['root_node'] = t.root
		active_x_i = list(range(self.p))
		# Note: 'lower_solve' returns the primal value and dual value, and it updates
		# t.root.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
		t.root.lower_solve(self.L0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		self.lower_bound = t.root.primal_value
		self.lower_bound_node_key = 'root_node'
		t.root.upper_solve(self.L0, self.l2, m=5)
		self.update_curr_best_int_sol(t.root, 'upper')
		self.optimality_gap = \
		    (self.curr_best_int_primal - self.lower_bound) / self.lower_bound

		#### Gather stats for 'observation'
		self.cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
		self.state['static_stats'] = get_static_stats(self.cov, self.x, self.y, \
		    	self.active_nodes, active_x_i, global_stats)

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
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, self.state['static_stats'], \
		self.batch_n, self.L0)
		search_q_hats = get_q_hats(self.search_model_name, search_stats, self.state['static_stats'], \
		self.batch_n, self.L0)

		# Record stats, keys and q_hats for the branch and search actions passed to the agent
		self.attach_action_option_stats(branch_stats, branch_keys, search_stats, search_keys, \
		    	branch_q_hats, search_q_hats)

		# Gather return values
		observation = np.concatenate([self.state['static_stats'], self.state['branch_option_stats'], \
		self.state['search_option_stats']])
		
		# prin(init_og = self.optimality_gap, init_ub = self.curr_best_int_primal, init_lb=self.lower_bound)
		return(observation)
    	#### End of reset() #######


	def get_info(self):
		info = {'node_count':self.node_counter,
			'branch_count': self.branch_counter,
			'search_count': self.search_counter,
			'step_count': self.step_counter,
			'x_file_name': self.x_file_name,
			'curr_best_primal_value': self.curr_best_int_primal,
			'beta': self.curr_best_int_beta,
			'support': self.curr_best_int_support,
			'lower_bound': self.lower_bound,
			'opt_gap': self.optimality_gap}
		return(info)

	def attach_action_option_stats(self, branch_stats, branch_keys, search_stats, search_keys, \
	branch_q_hats, search_q_hats):
		""" Attaches the stats, keys, and q_hats of the branch option and the search 
		option that are passed to the external agent.
		to the rl_env """
		# Select options to be passed to agent by taking arg max of q-hats
		# Note that when (random_number < greedy_epsilon) only one option is passed in,
		# and taking the arg max in that case is not strictly necessary.
		branch_ind = np.argmax(branch_q_hats) 
		self.state['branch_option_stats'] = branch_stats[branch_ind, :]
		self.state['branch_option_keys'] = branch_keys[branch_ind, :]
		self.state['branch_q_hat'] = branch_q_hats[branch_ind]
		search_ind = np.argmax(search_q_hats)
		self.state['search_option_stats'] = search_stats[search_ind, :]
		self.state['search_option_key'] = search_keys[search_ind]
		self.state['search_q_hat'] = search_q_hats[search_ind]
        
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

	def step(self, action):
		"""
		action: 0 or 1.    0 for branching, and 1 for searching
		"""
		self.step_counter += 1
		
		### Implement action
		if action == 1:
		    # search
		    branch_or_search = 'search'
		    self.search_counter += 1
		    search_node_key = self.state['search_option_key']
		    search_node = self.active_nodes[search_node_key]
		    search_node.searched = 1
		    search_support, search_betas = \
		    	get_search_solution(node=search_node, p=self.p, L0=self.L0, \
			l2=self.l2, y=self.y, batch_n=self.batch_n)

		    # Find primal value of search solution	
		    if search_support.shape[0] == 1:
		    	residuals = self.y - np.dot(self.x[:,search_support],    search_betas)
		    else:
		    	residuals = self.y - np.matmul(self.x[:,search_support], search_betas)
		    rss = np.dot(residuals, residuals)
		    search_sol_primal = rss/2 + self.L0*search_support.shape[0] + \
		        self.l2*np.dot(search_betas, search_betas)

		    # Check if new solution is best so far
		    if search_sol_primal < self.curr_best_int_primal:
		    	# Update current best integer solution
		    	self.curr_best_int_primal = search_sol_primal.copy()
		    	self.curr_best_int_beta = search_betas.copy()
		    	self.curr_best_int_support = search_support.copy()
		    	# Check all nodes for elimination
		    	for node_key in list(self.active_nodes.keys()):
		    		if self.active_nodes[node_key].primal_value > self.curr_best_int_primal:
		    		    del self.active_nodes[node_key]
		    #### End of Search routine
		    
		if action == 0:
		    # branch
		    branch_or_search = 'branch'
		    self.branch_counter += 1
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

		    # Solve relaxations in new nodes
		    self.active_nodes[node_name_1].lower_solve(\
		    	self.L0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)
		    self.active_nodes[node_name_2].lower_solve(\
		    	self.L0, self.l2, m=5, solver='l1cd', rel_tol=1e-4, mio_gap=0)

		    # Update current best integer solution (aka upper bound)
		    upper_bound_updated = False
		    # Check if node 1 relaxation solution is integer, and if so check if best so far
		    if int_sol(self.active_nodes[node_name_1], p=self.p, int_tol=self.int_tol, m=self.m):
		    	if self.active_nodes[node_name_1].primal_value < self.curr_best_int_primal:
		    		self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'primal')
		    		upper_bound_updated = True
		    	del self.active_nodes[node_name_1]
		    else:    # If not int., find node 1 upper bound, and check if best so far
		    	n_1_ub = self.active_nodes[node_name_1].upper_solve(self.L0, self.l2, m=5)
		    	if n_1_ub < self.curr_best_int_primal:
		    		self.update_curr_best_int_sol(self.active_nodes[node_name_1], 'upper')
		    		upper_bound_updated = True
		    # repeat for node 2
		    if int_sol(self.active_nodes[node_name_2], p=self.p, int_tol=self.int_tol, m=self.m):
		    	if self.active_nodes[node_name_2].primal_value < self.curr_best_int_primal:
		    		self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'primal')
		    		upper_bound_updated = True
		    	del self.active_nodes[node_name_2]
		    else:
		    	n_2_ub = self.active_nodes[node_name_2].upper_solve(self.L0, self.l2, m=5)
		    	if n_2_ub < self.curr_best_int_primal:
		    		self.update_curr_best_int_sol(self.active_nodes[node_name_2], 'upper')
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
		        self.lower_bound = min(self.active_nodes.values(), key=attrgetter('primal_value')).primal_value
		        self.lower_bound_node_key = reverse_lookup(self.active_nodes, \
		        min(self.active_nodes.values(), key=attrgetter('primal_value')))
		    # else: lower_bound remains unchanged
		    
		    ###### End of Branch routine ######
		    ###### End of Implementing Action ####
		

		### Calculate change in optimality gap 
		prev_opt_gap = self.optimality_gap
		self.optimality_gap = (self.curr_best_int_primal - self.lower_bound) / self.lower_bound
		change_in_opt_gap = prev_opt_gap - self.optimality_gap
		# Scaling for numerical reasons ... 
		# The un-scaled initial O.G. is often on the order of 10**-3, so when the model 
		# estimates zero, the mean squared error is often on the order of 10**6.
		change_in_opt_gap = change_in_opt_gap*10000
		
		### Attach stats on action taken <==> Update self.current_action
		if branch_or_search == 'branch':
		    specific_stats = self.state['branch_option_stats']
		    q_hat = self.state['branch_q_hat']
		    
		if branch_or_search == 'search':
		    specific_stats = self.state['search_option_stats']
		    q_hat = self.state['search_q_hat']
		    
		prev_action = self.current_action if self.step_counter > 1 else None
		
		self.current_action = action_taken(prev_action, branch_or_search, \
			    self.state['static_stats'], specific_stats, q_hat, self.step_counter, \
			    self.branch_counter, self.search_counter, \
			    change_in_opt_gap)
			    

		### If we're done ...
		if len(self.active_nodes) == 0:
		    ### Gather and save records
		    action = self.current_action
		    total_n_steps = action.step_number
		    total_n_branch = action.n_branch
		    total_n_search = action.n_search
		    
		    branch_action_records, search_action_records = [], []
		    branch_model_records, search_model_records = [], []
		    
		    # Get records of most recent action taken
		    action_record = np.concatenate([action.static_stats, \
		        action.specific_stats])
		    action_record = np.append(action_record, action.change_in_opt_gap)
		        
		    model_record = np.array([
		        action.step_number, action.n_branch, action.n_search, \
		        (total_n_steps - action.step_number), \
		        (total_n_branch - action.n_branch), \
		        (total_n_search - action.n_search), \
		        action.q_hat[0], \
		        action.change_in_opt_gap, \
		        self.batch_n], dtype = float)
		    
		    if action.branch_or_search == 'branch':
		    	branch_action_records.append(action_record)
		    	branch_model_records.append(model_record)
		    else:
		    	search_action_records.append(action_record)
		    	search_model_records.append(model_record)
    
		    # Get action records of previous actions taken
		    while action.prev_action is not None:
		    	action = action.prev_action
		    	action_record = np.concatenate([action.static_stats, \
		    	action.specific_stats])
		    	action_record = np.append(action_record, action.change_in_opt_gap)
		            
		    	model_record = np.array([
		    	    action.step_number, action.n_branch, action.n_search, \
		    	    total_n_steps - action.step_number, \
		    	    total_n_branch - action.n_branch, \
		    	    total_n_search - action.n_search, \
		    	    action.q_hat[0], \
		            action.change_in_opt_gap, \
		            self.batch_n], dtype = float)
		      
		    	if action.branch_or_search == 'branch':
		    		branch_action_records.append(action_record)
		    		branch_model_records.append(model_record)
		    	else:
		    		search_action_records.append(action_record)
		    		search_model_records.append(model_record)
		    		
		    # Save records
		    data_info = re.sub('x_', '', self.x_file_name)
		    data_info = re.sub('.npy', '', data_info)
		    log_L0 = -int(np.log10(self.L0))
		    data_info = data_info + 'L0_' +  str(log_L0)
		    
		    if branch_action_records:
		    	branch_action_records = np.vstack(branch_action_records)
		    	branch_record_dim = branch_action_records.shape[1]
		    	file_name = f'branch_action_rec_dim{branch_record_dim}_{data_info}'
		    	np.save(f'action_records/batch_{self.batch_n}/{file_name}', branch_action_records)

		    	branch_model_records = np.vstack(branch_model_records)
		    	branch_record_dim = branch_model_records.shape[1]
		    	model_data_info =  f'{data_info}_{self.branch_model_name}_b{self.batch_n}'
		    	file_name = f'branch_model_rec_dim{branch_record_dim}_{model_data_info}'
		    	np.save(f'model_records/batch_{self.batch_n}/{file_name}', branch_model_records)
		    	
		    if search_action_records:
		    	search_action_records = np.vstack(search_action_records)
		    	search_record_dim = search_action_records.shape[1]
		    	file_name = f'search_action_rec_dim{search_record_dim}_{data_info}'
		    	np.save(f'action_records/batch_{self.batch_n}/{file_name}', search_action_records)
		    	
		    	search_model_records = np.vstack(search_model_records)
		    	search_record_dim = search_model_records.shape[1]
		    	model_data_info =  f'{data_info}_{self.search_model_name}_b{self.batch_n}'
		    	file_name = f'search_model_rec_dim{search_record_dim}_{model_data_info}'
		    	np.save(f'model_records/batch_{self.batch_n}/{file_name}', search_model_records)
		    
		    info = self.get_info()
		    # file_name = f'ep_res_records/batch_{self.batch_n}/{data_info}.pkl' 
		    # Drop this if tensor Board works
		    # with open(file_name, 'wb') as f:
		        # pickle.dump(info, f)

		    ### Gather return values
		    done = True
		    observation = np.zeros((65))
		    reward = change_in_opt_gap
	
		    return(observation, reward, done, info)
		### End of "If we're done" #########
		    
		    
		### If we're NOT done . . . 
		### Gather Stats
		global_stats = np.array([self.L0, self.l2, self.p])		
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

		# Get q_hats applying models 
		# (to possilby 1 action option, all action options, or a capped # of action options)
		branch_q_hats = get_q_hats(self.branch_model_name, branch_stats, self.state['static_stats'], \
		self.batch_n, self.L0)
		search_q_hats = get_q_hats(self.search_model_name, search_stats, self.state['static_stats'], \
		self.batch_n, self.L0)
		# Attach stats, keys and q_hats for the branch and search actions passed to the agent
		self.attach_action_option_stats(branch_stats, branch_keys, search_stats, search_keys, \
		    branch_q_hats, search_q_hats)
		
		# Gather return values
		observation = np.concatenate([self.state['static_stats'], self.state['branch_option_stats'], \
		    self.state['search_option_stats']])
		done = False
		reward = change_in_opt_gap
		# prin(reward=reward, prev_opt_gap = prev_opt_gap, og = self.optimality_gap, \
		#    ub = self.curr_best_int_primal, lb=self.lower_bound)
		info = self.get_info()
		return(observation, reward, done, info)

	def close(self):
    		pass
