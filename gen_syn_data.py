# Toby DeKara
# Created: Oct 19, 2021
# Last edited: Oct 19, 2021
# First draft of a python script to generate synthetic data sets using the L0bnb function gen_synthetic

import numpy as np
from l0bnb import gen_synthetic
import os
path = 'data/tdekara/synthetic_data'
for i in range(5):
	X, y, b = gen_synthetic(n=5, p=5, supp_size=1)
	np.save(os.path.join(path,'X_{}'.format(i)), X)
	np.save(os.path.join(path,'y_{}'.format(i)), y)
	np.save(os.path.join(path,'b_{}'.format(i)), b)

