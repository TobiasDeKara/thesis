#!/bin/bash

# --- Start of slurm commands -----------

# Runtime notes
# p=1000, batches of n_mat=1000, took 1:05:00 and 1.2G
#SBATCH --time=1:30:00

##SBATCH -N 1
##SBATCH -n 5

#SBATCH --mem=2G
#SBATCH --array=0-95

# Specify a job name:
#SBATCH -J gen_p3

# Specify an output file
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/gen_syn_%J_%a.err
#SBATCH -o ./sbatch_outputs/gen_syn_%J_%a.out
#----- End of slurm commands ----

# Run commands
source ~/L0_env_p3.9.0/bin/activate

# python -u ~/thesis/gen_syn_data.py validation $SLURM_ARRAY_TASK_ID
python -u ~/thesis/gen_syn_data.py $SLURM_ARRAY_TASK_ID

# Format: python -u ~/thesis/gen_syn_data.py <run_n> <batch_n>
# Each call to gen_syn_data.py produces 1 batch of data, and sends
# one 'seed_support_array' to the <run_n> directory that has the seed # for each set of 
# inputs and the true support.
# 'u' for unbuffered, meaning print statements are returned immediately

