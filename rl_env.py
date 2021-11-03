# Here I create a root node using the 'Node' class from l0bnb,
# and then use 'solve' to solve the integer relaxation.
# 
# Much of the code below is taken directly from l0bnb, 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import os
import subprocess
import numpy as np
from l0bnb.node import Node

x_file_list = subprocess.run("cd synthetic_data; ls X* -1", capture_output=True, text=True, shell=True).stdout.splitlines()
y_file_list = subprocess.run("cd synthetic_data; ls y* -1", capture_output=True, text=True, shell=True).stdout.splitlines()
b_file_list = subprocess.run("cd synthetic_data; ls b* -1", capture_output=True, text=True, shell=True).stdout.splitlines()

# load synthetic data
x = np.load(os.path.join('synthetic_data', 'X_gen_syn_n3_p3_supp10_seed10002539.npy'))
y = np.load(os.path.join('synthetic_data', 'y_gen_syn_n3_p3_supp10_seed10002539.npy'))

# Create node_0
p = x.shape[0]
zlb = np.zeros(p, dtype='int32')
zub = np.ones(p, dtype='int32')
node_0 = Node(parent = None, zlb = zlb, zub = zub, x = x, y = y)
node_0.xi_norm = np.linalg.norm(node_0.x, axis=0) ** 2

# Solve relaxation
# This returns the primal value and dual value, and it updates
# self.primal_value, .dual_value, .primal_beta, .z, .support, .r, .gs_xtr, and .gs_xb
# (I'm not sure what 'rel_tol' is, or what a good value is.)
node_0.lower_solve(l0=0, l2=0, m=10, solver='l1cd', rel_tol=1, mio_gap=0)

# TODO: turn this into a state space
# TODO: handle branching

