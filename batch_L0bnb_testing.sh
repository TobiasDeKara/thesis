#!/bin/bash

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=1:00:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=4G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J p1_p2_max_frac

#SBATCH --array=0-8

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/L0_test_%J_%a.err
#SBATCH -o ./sbatch_outputs/L0_test_%J_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/L0bnb_testing.py 50 $SLURM_ARRAY_TASK_ID
python -u ~/thesis/L0bnb_testing.py 100 $SLURM_ARRAY_TASK_ID

# python -u ~/thesis/L0bnb_testing.py {p} {L0_L2_n} 
# 'u' for unbuffered, meaning print statements are returned immediately
