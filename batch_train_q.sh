#!/bin/bash

# --- Start of slurm commands -----------
# Run time notes:
# 20 hrs, at least 700 epochs, 3 and 4 layers
# 500 epochs, 5 and 6 layers, 13-18 hours

#SBATCH --time=40:00:00
#SBATCH --mem=3G 

# Specify a job name:
#SBATCH -J big_q_tr

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
python ~/thesis/train_q_models.py $1 $2 $3 $4 $5

# <run_n> <n_layer> <drop_out>   <regularization> <learning_rate>
# {all, n}  {3,4}    {yes, no}    {True, False}    default = 0.001, 0.00001, 1e-05


