#!/bin/bash

# --- Start of slurm commands -----------

#SBATCH --time=6:00:00

##SBATCH -p debug

#SBATCH --mem=8G

#SBATCH --array=0-9

# Specify a job name:
#SBATCH -J perm_imp

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/perm_imp_%J.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#SBATCH -e ./sbatch_errors/perm%J_%a.err
#SBATCH -o ./sbatch_outputs/perm%J_%a.out
#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
# python -u ~/thesis/permutation_imp.py $SLURM_ARRAY_TASK_ID binary
python -u ~/thesis/permutation_imp.py $SLURM_ARRAY_TASK_ID sig_num
