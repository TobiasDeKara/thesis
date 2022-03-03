#!/bin/bash

# Run times:
# 3k steps 30-50 min, less than 1G
# 3k steps x 100 batches x 3 L0 values = ~ 3 hrs


# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=1:20:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=2G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J run_rl_3k

#SBATCH --array=100-111   # 0-287  # 0- 3*n_batches-1

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J_%a.err
#SBATCH -o ./sbatch_outputs/%J_%a.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/run_rl.py $1 $SLURM_ARRAY_TASK_ID

# python -u ~/thesis/run_rl.py {run_n} {batch_L0_n}
# batch_n = int(batch_L0_n/3)
# -log(L0) = (batch_L0_n % 3) + 2 i.e. cycles through 2,3,4
# 'u' for unbuffered, meaning print statements are returned immediately

