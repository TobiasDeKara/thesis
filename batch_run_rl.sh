#!/bin/bash

# Run times:
# p3, 1000 time steps, 2 hours, Max 2.1G, 5 cores, dummy Vec
# p3, 1000 time steps, 2 hours, Max 2.2G, 2 cores, dummy Vec
# p3, 5000 time steps, after 2.5 hrs out of memory at 3G, 1 core, dummy Vec
# p3, 1000 time steps, 1.5 hours, Max 2.2.G, 1 core, dummy Vec
# p3, 1500 time steps, 2:15, Max 3G, 1 core, dummy Vec
# p3, 1500 time steps, 2:15, Max 6G, 1 core, dummy Vec, with model records
# p3, 1500 time steps, 1.5-2 hrs, Max 4G, 1 core, dummy Vec, deleting action recs and model recs
# p3, 1500 time steps, 1 hr, Max 3G, 1 core, dummy Vec, deleting action recs and model recs, & more?
# p3, 2000 time steps, 3-4 hrs, Max 8G, 1 core, dummy Vec
# p3, 3000 time steps, 6-7 hrs, Max 17G, 1 core, dummy Vec

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=3:00:00

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=8G # Note: by default the memory is per NODE not per CPU

# Specify a job name:
#SBATCH -J p3_1.5k

#SBATCH --array=0-95:8

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J_%a.err
#SBATCH -o ./sbatch_outputs/%J_%a.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python ~/thesis/run_rl.py $1 $SLURM_ARRAY_TASK_ID
# python ~/thesis/run_rl.py validation 96

# python -u ~/thesis/run_rl.py {run_n} {first batch number}
# Note: python -u ~/thesis/run_rl.py <batch_n>, will make record, 
# 	and search param/result sub-dir. batch_n
# 'u' for unbuffered, meaning print statements are returned immediately

