# Here I create a root node using the 'Node' class from l0bnb,
# and then use 'solve' to solve the integer relaxation.
# 
# Much of the code below is taken directly from l0bnb, 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import os
import subprocess
import numpy as np
from l0bnb.node import Node
from l0bnb import BNBTree as tree

# x_file_list = subprocess.run("cd synthetic_data; ls X* -1", capture_output=True, text=True, shell=True).stdout.splitlines()
# y_file_list = subprocess.run("cd synthetic_data; ls y* -1", capture_output=True, text=True, shell=True).stdout.splitlines()
# b_file_list = subprocess.run("cd synthetic_data; ls b* -1", capture_output=True, text=True, shell=True).stdout.splitlines()

# load synthetic data
x = np.load(os.path.join('synthetic_data', 'X_gen_syn_n3_p3_supp10_seed10002539.npy'))
y = np.load(os.path.join('synthetic_data', 'y_gen_syn_n3_p3_supp10_seed10002539.npy'))

t = tree(x, y)
l0=1
l2=1
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
# print(f'relax_sol: {relax_sol}')

# TODO: turn this into a state space




# TODO: handle branching
