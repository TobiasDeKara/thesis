import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import os
import re

# objective = 1/2 RSS + L0 * sum(z_i) + L2 * sum(s_i)  
# 1/2 RSS = 1/2 beta^T x^T x beta - y^T x beta + 1/2 y^T y

# Constraints:  
# beta_i <= s_i * z_i  
# -M z_i <= beta_i <= M z_i  
# s_i >= 0  
# z_i in {0, 1}

def gp_L0_L2(x, y, L0, L2, big_M=5, verbose=False):
    n, p = x.shape
    assert n == y.shape[0]
    
    model = gp.Model()
    beta = model.addVars(p, lb=-GRB.INFINITY, name='beta')
    z = model.addVars(p, vtype=GRB.BINARY, name='z')
    s = model.addVars(p, lb=0, name='s')

    # Objective
    Quad = np.matmul(x.T, x)
    lin = np.matmul(y.T, x)
    # We can't use matmul below, because 'beta' is a 'tupledict'
    obj = 0.5 * sum(Quad[i,j] * beta[i] * beta[j]
              for i, j in product(range(p), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(p))
    obj += 0.5 * np.dot(y, y)
    obj += L0*sum(z[i] for i in range(p))
    obj += L2*sum(s[i] for i in range(p))
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    for i in range(p):
        model.addConstr(-big_M*z[i] <= beta[i], f'big_M_lower_{i}')
        model.addConstr(beta[i] <= big_M*z[i], f'big_M_upper_{i}')
        model.addConstr(beta[i]**2 <= s[i]*z[i], f'perspective_cone_{i}')
    
    if not verbose:
        model.params.OutputFlag = 0
        
    model.params.mipgap = 1e-2   # L0BnB defaults to 1e-2, I passed 0 in 'rl_env.py', but with rel_tol
    model.params.IntFeasTol = 1e-4 # L0BnB uses 1e-4 or 1e-6, I used 1e-4
    model.params.Presolve = 0
        
    model.optimize()
    
    n_nodes = model.NodeCount
    
    coefs = np.array([beta[i].X for i in range(p)])
    # print(coefs)
    supp_size = sum(np.abs(coefs) > 0)
    
    # return model
    return [n_nodes, supp_size]
  
  log_p = 1
p = 50

path = f'./Desktop/Thesis/synthetic_data/p{log_p}/batch_1/'
x_file_list = [f for f in os.listdir(path) if re.match('x', f)]

gb_res_rec_list = []

for x_file_name in x_file_list:
    snr = re.search('snr(\d*)', x_file_name)[1]
    corr = re.search('corr(\d.\d)', x_file_name)[1]
    
    y_file_name = x_file_name.replace('x', 'y')
    
    x = np.load(f'{path}{x_file_name}')
    y = np.load(f'{path}{y_file_name}').reshape(-1,)
    
    for L0 in [10**-2, 10**-3, 10**-4]:
        for L2 in [10**-2, 10**-3, 10**-4]:
            
            n_nodes, supp_size = gp_L0_L2(x, y, L0, L2, verbose=False)
    
            res_rec = np.array([p, L0, L2, corr, snr, n_nodes-1, supp_size, 'gurobi', 'None'])
    
            gb_res_rec_list.append(res_rec)
    
    
gb_res_rec_p1 = np.vstack(gb_res_rec_list)

supp_size = gb_res_rec_p1[:,6].astype(float)
supp_size.mean()

n_steps = gb_res_rec_p1[:,5].astype(float)
n_steps.mean()
