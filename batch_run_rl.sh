#!/bin/bash

# Run times:
# 3k steps 30-50 min, less than 1G
# 3k steps x 100 batches x 3 L0 values = ~ 3 hrs
# 9k steps x 100 batches x 3 L0 values = 16 hrs

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=4:20:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=2G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J run_rl_900_ep

#SBATCH --array=0-8  # 0- 3*n_batches-1

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J_%a.err
#SBATCH -o ./sbatch_outputs/%J_%a.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/run_rl.py $1 $2 $SLURM_ARRAY_TASK_ID

# python -u ~/thesis/run_rl.py {p} {run_n} {L0_L2_n}
# batch_n = 0
# log_L0 = int(L0_L2_n/3) + 2
# log_L2 = (L0_L2_n % 3) + 2    i.e. log_L0 and log_L2 each cycle through 2,3,4
# 'u' for unbuffered, meaning print statements are returned immediately

