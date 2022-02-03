#!/bin/bash

# Run times:
# subprocVec, 24 cores; 48G; 50,000 total time steps; p=5; runtime 2:20:00, not sure what L0 was
# subprocVec, 24 cores; 48G; 50,000 total time steps; p=5, L0 in {10**-2, 10**-3, 10**-4},...
# 	 TIME OUT at 4:10:00, but it did 90,000 steps!? 
# dummyVec, 24 cores, 48G, 10,000 total time steps, p=t, L0 as above, 
# 	TIME OUT at 2:00:00 but it did 14,000 steps!?
# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=5:00:00

##SBATCH -p debug

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=2G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J rl_4

#SBATCH --array=0-12

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J%a.err
#SBATCH -o ./sbatch_outputs/%J%a.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python ~/thesis/run_rl.py $SLURM_ARRAY_TASK_ID

# python -u ~/thesis/run_rl.py {first batch number} {run_n}
# Note: python -u ~/thesis/run_rl.py <batch_n>, will make record, 
# 	and search param/result sub-dir. batch_n
# 'u' for unbuffered, meaning print statements are returned immediately

