# Toby DeKara
# Created: Oct 19, 2021
# Last edited: Oct 26, 2021
# A python script to generate synthetic data sets using the L0bnb function gen_synthetic

# TODO: look at articles to find the number of matrices they used

n = 3 # the number of rows in the matrix X, and entries in vector y
p = 5 # the number of columns of the matrix X, and entries of vector b
supp_size = 1 # the number of columns of X used to construct y
k = 3 # the number of matrices X, and vectors y to construct

import numpy as np
from l0bnb import gen_synthetic
import os
path = 'synthetic_data' # this is a symbolic link to /users/tdekara/data/tdekara/synthetic_data

for i in range(k):
	seed = n*p*i*10
	X, y, b = gen_synthetic(n=n, p=p, supp_size=supp_size, seed=seed)
	X_centered = X - np.mean(X, axis = 0)
	X_normalized =  X_centered / np.linalg.norm(X_centered, axis = 0)
	# Note that the norm in 'np.linalg.norm'(in the line above) defaults to L2
	y_centered = y - np.mean(y)
	y_normalized = y_centered / np.linalg.norm(y_centered)
	
	filetag = f'_gen_syn_n{n}_p{p}_supp{supp_size}_seed{seed}'
	
	np.save(os.path.join(path,'X' + filetag), X_normalized)
	np.save(os.path.join(path,'y' + filetag), y_normalized)
	if i == 0:
		np.save(os.path.join(path,'b' + filetag), b)

