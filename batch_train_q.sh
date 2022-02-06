#!/bin/bash

# --- Start of slurm commands -----------

# Run times:
# 2,000 epochs, of run_0 data, for branch model, 2:50:00, 2G, with histograms
# 2,000 epochs, of run_0 data, for search model, 28 min, 2G, no histograms

# 2,000 epochs, of run_1 data, for branch model, 	2G, no histograms
# 2,000 epochs, of run_1 data, for search model, 45 min, 2G, no histograms

#SBATCH --time=1:30:00
#SBATCH --mem=2G 

# Specify a job name:
#SBATCH -J train_q_search

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
python ~/thesis/train_q_models.py $1


