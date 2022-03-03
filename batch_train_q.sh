#!/bin/bash

# --- Start of slurm commands -----------

#SBATCH --time=1:00:00
#SBATCH --mem=2G 

# Specify a job name:
#SBATCH -J q_tr_p3

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
python ~/thesis/train_q_models.py $1 $2


