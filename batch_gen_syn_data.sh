#!/bin/bash

# --- Start of slurm commands -----------

# Runtime notes
# p=1000, batches of n_mat=1000, took 1:05:00 and 1.2G

#SBATCH --time=2:00:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 5

#SBATCH --mem=4G
##SBATCH --array=0-100

# Specify a job name:
#SBATCH -J gen_syn

# Specify an output file
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/gen_syn_%J.err
#SBATCH -o ./sbatch_outputs/gen_syn_%J.out
#----- End of slurm commands ----

# Run commands
source ~/L0_env_p3.9.0/bin/activate
python -u ~/thesis/gen_syn_data.py $1 $2 $3 $4
# python -u ~/thesis/gen_syn_data.py $SLURM_ARRAY_TASK_ID $1 $2 $3

# Format: python -u ~/thesis/gen_syn_data.py <batch_n> <p> <rho (correlation)> <SNR>
# Each call to gen_syn_data.py produces 1 batch of data, and sends
# one 'seed_support_array' to synthetic_data/{p_sub_dir}/{batch_n},
#  that has the seed and the true support.

# 'u' for unbuffered, meaning print statements are returned immediately

