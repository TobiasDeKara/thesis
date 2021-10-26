# Toby DeKara
# Created: Oct 19, 2021
# Last edited: Oct 26, 2021
# A python script to generate synthetic data sets using the L0bnb function gen_synthetic

# Citation: This work is based on the work described in 
# Hazimeh, Mazumber and Saab "Sparse Regression at Scale" (preprint as of 2021) 
# The L0bnb package was written by Hazimeh and Saab and is available on github

# Variable descriptions:
# 'n' is the number of rows in the matrix X, and entries in vector y
# 'p' is the number of columns of the matrix X, and entries of vector b
# 'supp_size' is the number of columns of X used to construct y
# 'k' is the number of matrices X, and vectors y to construct
# 'rho' is the amount of constant correlation between all columns of X
# 'snr' is the signal to noise ratio (See 

# Variable settings: 
# Hazimeh et al.  use: 
# supp_size = 10, rho = 0.1, n = 10^3, p = {10^3, 10^4, 10^5, 10^6}, snr = 5

n = int(10**3) 
p = int(10**4) 
supp_size = 10 
k = 1000 

import numpy as np
from l0bnb import gen_synthetic
import os
import time

path = 'synthetic_data' # this is a symbolic link to /users/tdekara/data/tdekara/synthetic_data

for i in range(k): 
	seed = int(10**8 * np.random.random_sample())
	X, y, b = gen_synthetic(n=n, p=p, supp_size=supp_size, rho = 0.1, snr = 5, seed=seed)
	X_centered = X - np.mean(X, axis = 0)
	X_normalized =  X_centered / np.linalg.norm(X_centered, axis = 0)
	# Note that the norm in 'np.linalg.norm'(in the line above) defaults to L2
	y_centered = y - np.mean(y)
	y_normalized = y_centered / np.linalg.norm(y_centered)
	
	# Note: For brevity, the n and p values recorded in the file names are the log base 10 of the actual values
	filetag = f'_gen_syn_n{int(np.log10(n))}_p{int(np.log10(p))}_supp{supp_size}_seed{seed}'
	np.save(os.path.join(path,'X' + filetag), X_normalized)
	np.save(os.path.join(path,'y' + filetag), y_normalized)
	filetag_b = f'_gen_syn_n{int(np.log10(n))}_p{int(np.log10(p))}_supp{supp_size}'
	if i  == 0:
		np.save(os.path.join(path,'b' + filetag_b), b)


# Measuring run time
# n = 10^3, p = 5, k = 3, runtime = 10s, MAXRSS = 2.6 Mb
# n = 10^3, p = 10^3, k = 3, runtime = 7s , MAXRSS = 2.6 Mb
# n = 10^3, p = 10^3, k = 1000, runtime = 2 min, MAXRSS = .12 Gb
# n = 10^3, p = 10^4, k = 1000, runtime = , MaxRSS = 
