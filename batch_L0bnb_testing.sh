#!/bin/bash

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=40:00:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=32G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J L0bnb_test

##SBATCH --array=0-8

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/L0_test_%J.err
#SBATCH -o ./sbatch_outputs/L0_test_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/L0bnb_testing.py $1

# python -u ~/thesis/L0bnb_testing.py {p}
# 'u' for unbuffered, meaning print statements are returned immediately
