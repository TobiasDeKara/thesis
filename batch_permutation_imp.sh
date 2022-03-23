#!/bin/bash

# --- Start of slurm commands -----------

#SBATCH --time=6:00:00

##SBATCH -p debug

#SBATCH --mem=2G

#SBATCH --array=1-9

# Specify a job name:
#SBATCH -J perm_imp


# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/perm_imp_%J.err

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
python -u ~/thesis/permutation_imp.py $SLURM_ARRAY_TASK_ID

