#!/bin/bash

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=2:00:00

#SBATCH -N 1
#SBATCH -n 5

#SBATCH --mem=2G

# Specify a job name:
#SBATCH -J gen_p3_r0

# Specify an output file
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/gen_syn_%J.err
#SBATCH -o ./sbatch_outputs/gen_syn_%J.out
#----- End of slurm commands ----

# Run commands
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/gen_syn_data.py 0
# python -u ~/thesis/gen_syn_data.py <run_n>
# Each call to gen_syn_data.py produces 96 batches of data, and
# one 'seed_support_array' that has the seed # for each set of 
# inputs and the true support.
# 'u' for unbuffered, meaning print statements are returned immediately

