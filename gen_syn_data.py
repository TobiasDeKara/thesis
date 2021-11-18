# Toby DeKara
# Created: Oct 19, 2021
# A python script to generate synthetic data sets
# Much of this script is taken directly from  'gen_synthetic',
# in the package 'l0bnb', which is under an MIT License, 
# copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab].

# Variable settings: 
# Hazimeh et al.  use: 
# supp_size = 10, rho = 0.1, n = 10^3, p = {10^3, 10^4, 10^5, 10^6}, snr = 5

import numpy as np
from numpy.random import normal
import os

def make_syn_data(n=10**3, p=10**3, supp_size=10, rho=0.1, snr=5):
    """Generate a synthetic regression dataset.
    The data matrix x is sampled from a multivariate gaussian and
    the error term epsilon is sampled from an normal distribution (independent
    of x). The response y = xb + epsilon, where b is a sparse
    vector, where all the nonzero elements are set to 1.
    Inputs:
        n: Number of samples, i.e. number of rows of matrix x, and length of y.
        p: Number of features, i.e. number of columns of x.
        supp_size: Number of non-zero entries in b. This is number of columns 
		of x used to construct y.
        rho: Constant correlation parameter. The same random vector is added
		to all columns of x so that the mean correlation between
		column of x is rho.
        snr: Signal-to-noise ratio.
        seed: Numpy seed.
    Returns:
        x: The data matrix.
        y: The response vector.
        b: The true vector of regression coefficients.
    """
    seed = int(10**8 * np.random.random_sample())
    np.random.seed(seed)
 
    # Make b
    b = np.zeros(p)
    support = np.random.choice(range(p), size=supp_size)    
    b[support] = np.ones(supp_size)
    
    # Make x
    x = normal(size=(n, p)) + np.sqrt(rho / (1 - rho)) * normal(size=(n, 1))
    x_centered = x - np.mean(x, axis = 0)
    x_normalized =  x_centered / np.linalg.norm(x_centered, axis = 0)
    # Note that the norm in 'np.linalg.norm'(in the line above) defaults to L2
    
    # Make y
    mu = x.dot(b)
    var_xb = (np.std(mu, ddof=1)) ** 2
    sd_epsilon = np.sqrt(var_xb / snr)
    epsilon = normal(size=n, scale=sd_epsilon)
    y = mu + epsilon
    y_centered = y - np.mean(y)
    y_normalized = y_centered / np.linalg.norm(y_centered)

    # Save 
    path = 'synthetic_data' 
    # Note: this is a symbolic link to /users/tdekara/data/tdekara/synthetic_data,
    #     So this is all saved in 'data' not the home directory.
    # Note: For brevity, the n and p values recorded in the file names are 
    # the log base 10 of the actual values

    # Changed for mini data
    filetag = f'_gen_syn_n{int(np.log10(n))}_pmini_supp{supp_size}_seed{seed}' 
   
    #filetag = f'_gen_syn_n{int(np.log10(n))}_p{int(np.log10(p))}_supp{supp_size}_seed{seed}'  
    np.save(os.path.join(path,'b' + filetag), b)
    np.save(os.path.join(path,'x' + filetag), x_normalized)
    np.save(os.path.join(path,'y' + filetag), y_normalized)
    del x, y, b

for i in range(10):
    make_syn_data(p=5)


