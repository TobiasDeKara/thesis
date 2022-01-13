# Toby DeKara
# Created: Oct 19, 2021
# A python script to generate synthetic data sets
# This script is adapted from  'gen_synthetic',
# in the package 'l0bnb', which is under an MIT License, 
# copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab].

# Variable settings: 
# Hazimeh et al.  use: 
# supp_size = 10, rho = 0.1, n = 10^3, p = {10^3, 10^4, 10^5, 10^6}, snr = 5

import numpy as np
from numpy.random import multivariate_normal, normal
import os

make_syn_data(n_mat=1000, p=5, supp_size=1)
    
def make_syn_data(n_mat=10**3, n=10**3, p=10**3, supp_size=10, rho=0.5, snr=5):
    """Generate a synthetic regression dataset: y, x, and b.
    The data matrix x is sampled from a multivariate gaussian with exponential 
    correlation between columns.
    The response y = xb + epsilon, where b is a vector with 'supp_size' 
    randomly chosen entries equal to 1 and the rest equal to 0.
    The error term epsilon is sampled from an normal distribution (independent
    of x). 
    
    Inputs:
        n: Number of samples, i.e. number of rows of matrix x, and length of y.
        p: Number of features, i.e. number of columns of x.
        supp_size: Number of non-zero entries in b. This is number of columns 
		of x used to construct y.
        rho: Exponential correlation parameter. cov_mat[row, col] = rho**np.abs(row-col)
        snr: Signal-to-noise ratio.
        seed: Numpy seed.
    Returns:
        x: The data matrix.
        y: The response vector.
        b: The true vector of regression coefficients.
    """
    for _ in range(n_mat):
        seed = int(10**8 * np.random.random_sample())
        np.random.seed(seed)
     
        # Make b
        b = np.zeros((p,1))
        support = np.random.choice(range(p), size=supp_size)    
        b[support] = np.ones(supp_size)
        
        # Make x
        cov_mat = np.zeros((p,p))
        for row in range(p):
            for col in range(p):
                cov_mat[row, col] = rho**np.abs(row-col)
        
        x = multivariate_normal(mean=np.zeros(p), cov=cov_mat, size=n)
        x_centered = x - np.mean(x, axis = 0)
        x_normalized =  x_centered / np.linalg.norm(x_centered, axis = 0)
        # Note that the norm in 'np.linalg.norm'(in the line above) defaults to L2
        
        # Make y
        mu = np.matmul(x, b)
        var_xb = (np.std(mu, ddof=1)) ** 2
        sd_epsilon = np.sqrt(var_xb / snr)
        epsilon = normal(size=n, scale=sd_epsilon)
        y = mu + epsilon.reshape(-1,1)
        y_centered = y - np.mean(y)
        y_normalized = y_centered / np.linalg.norm(y_centered)
    
        # Save 
        path = 'synthetic_data' 
        # Note: this is a symbolic link to /users/tdekara/data/tdekara/synthetic_data,
        #     So this is all saved in 'data' not the home directory.
        
        # Note: For brevity, the n and p values recorded in the file names are 
        # the log base 10 of the actual values
    
        # Make file name
        if p == 5:
            # Changed for mini data because we don't want to take the log of p=5
            filetag = f'_gen_syn_n{int(np.log10(n))}_pmini_supp{supp_size}_seed{seed}' 
        else:
            filetag = f'_gen_syn_n{int(np.log10(n))}_p{int(np.log10(p))}_supp{supp_size}_seed{seed}'  
        
        np.save(os.path.join(path,'b' + filetag), b)
        np.save(os.path.join(path,'x' + filetag), x_normalized)
        np.save(os.path.join(path,'y' + filetag), y_normalized)
        del x, y, b


# For Testing
# import subprocess
# x_file_list = subprocess.run( \
# 		    	f"cd synthetic_data; ls x*_pmini_* -1U", \
# 		    	capture_output=True, text=True, shell=True).stdout.splitlines()
# x = x_file_list[3]
# test = np.load(os.path.join(path, x))
# test.shape
# np.cov(test.T)

