# Toby DeKara
# Created: Oct 19, 2021
# A python script to generate synthetic data sets
# This script is adapted from  'gen_synthetic',
# in the package 'l0bnb', which is under an MIT License, 
# copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab].

# Variable settings: 
# Bertsimas et al.  use: 
# supp_size in {5,10}, rho in {0.5, 0.8, 0.9}?
# Hazimeh et al, use n = 10^3, p = {10^3, 10^4, 10^5, 10^6}, snr = 5

import sys
import numpy as np
from numpy.random import multivariate_normal, normal
import os
import subprocess

def make_syn_data(n_mat=10**3, n=10**3, p=int(sys.argv[2]), rho=float(sys.argv[3]), \
	snr=float(sys.argv[4]), batch_n = int(sys.argv[1])):
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
    supp_size = int(0.1*p)
    # Create n_th batch sub-directory
    if p == 5:
        p_sub_dir = 'pmini'
    else: 
        p_sub_dir = f'p{int(np.log10(p))}'

    xy_out_dir = f'synthetic_data/{p_sub_dir}/batch_{batch_n}'
    os.makedirs(xy_out_dir, exist_ok=True)

    seed_support_list = []

    for _ in range(n_mat):
        seed = int(10**9 * np.random.random_sample())
        np.random.seed(seed)
     
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
        unshuffled_support = [i for i in range(p) if i % (p/supp_size) == 0]
        # unshuffled_support = np.random.choice(5, 2, replace=False)
        b = np.zeros((p, 1))
        b[unshuffled_support] = np.ones((supp_size,1))
        mu = np.matmul(x, b)
        var_xb = (np.std(mu, ddof=1)) ** 2
        sd_epsilon = np.sqrt(var_xb / snr)
        epsilon = normal(size=n, scale=sd_epsilon)
        y = mu + epsilon.reshape(-1,1)
        y_centered = y - np.mean(y)
        y_normalized = y_centered / np.linalg.norm(y_centered)
    
        # Shuffle x
        perm = np.random.permutation(p)
        x_shuffled = x_normalized[:, perm]

        # Identify support after shuffle
        support = [ind for ind in range(p) if perm[ind] in set(unshuffled_support)]

        # Record support
        seed_support = np.concatenate([np.array(seed, ndmin=1), np.array(support, ndmin=1)])
        seed_support_list.append(seed_support)

        # Save
        # Note: For brevity, the n and p values recorded in the file names are 
        # the log base 10 of the actual values
    
        # Make file name
        filetag = f'gen_syn_n{int(np.log10(n))}_{p_sub_dir}_corr{rho}_snr{snr}_seed{seed}' 
       
        np.save(f'{xy_out_dir}/x_{filetag}', x_shuffled)
        np.save(f'{xy_out_dir}/y_{filetag}', y_normalized)
        del x, y
    
    seed_support_array = np.vstack(seed_support_list)
    b_out_dir = f'synthetic_data/{p_sub_dir}/seed_support_records'
    os.makedirs(b_out_dir, exist_ok=True)
    np.save(f'{b_out_dir}/seed_support_record_corr{rho}_snr{snr}_batch_{batch_n}', seed_support_array)

if __name__ == "__main__":
	make_syn_data(n_mat=10)

