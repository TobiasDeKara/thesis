#!/bin/bash

# Run times:
# The memory usage is a function of the steps per ep
# p = 50, 2.5 G max, ~400 steps per hour, working same episode for 8 hours
# p = 50, 10k steps (100 ep x 100 steps per ep) 3 hours, less than 1G
# p = 50, 81k steps (810 ep total (90 per array id) x 100 steps per ep) 3 hours, less than 1G
# p = 50, 30k steps (100 ep x 300 steps per ep) 13 hours, less than 1G
# p = 100 10k steps (100 ep x 100 steps per ep) 5 hours
# p = 1000 10k steps (100 ep x 100 steps per ep) 34 hours
# p = 1000 2k steps (100 ep x 20 steps per ep) 3 hours


# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=6:00:00 # test p50 6hrs, test p100 10hrs, test 1k ??

##SBATCH -p debug

##SBATCH -N 1
##SBATCH -n 2
#SBATCH --mem=2G 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#SBATCH --array=0-8  # 0- 3*n_batches-1

# Specify a job name:
#SBATCH -J te_50_bin_ran
##SBATCH -J te_50_bb
##SBATCH -J te_50_sig_num
##SBATCH -J te_100_bin_ran
##SBATCH -J te_100_bb
##SBATCH -J te_100_sig_num
##SBATCH -J te_1000_bin_ran
##SBATCH -J te_1000_bb
##SBATCH -J te_1000_sig_num
##SBATCH -J te_100_bin_spec



# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J_%a.err
#SBATCH -o ./sbatch_outputs/%J_%a.out

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

### Train
#  {p} {run_n} {batch_n} {L0_L2_n} {max_n_steps} {test_train} {model_scope} {reward_format}

# The following parmas were estimated to run in 10 hours
# python ~/thesis/run_rl.py   50    4    2       $SLURM_ARRAY_TASK_ID  250 train range binary
# python ~/thesis/run_rl.py  100    4    2       $SLURM_ARRAY_TASK_ID  100 train range binary
# python ~/thesis/run_rl.py 1000    4    2       $SLURM_ARRAY_TASK_ID  50 train range binary

### Test
python -u ~/thesis/run_rl.py   50    98    1   $SLURM_ARRAY_TASK_ID  50 test range binary
# python -u ~/thesis/run_rl.py   50    98    1   $SLURM_ARRAY_TASK_ID  50 test range big_binary
# python -u ~/thesis/run_rl.py   50    98    1   $SLURM_ARRAY_TASK_ID  50 test range sig_num

# python -u ~/thesis/run_rl.py   100   98    1   $SLURM_ARRAY_TASK_ID  100 test range binary
# python -u ~/thesis/run_rl.py   100   98    1   $SLURM_ARRAY_TASK_ID  100 test range big_binary
# python -u ~/thesis/run_rl.py   100   98    1   $SLURM_ARRAY_TASK_ID  100 test range sig_num

# python -u ~/thesis/run_rl.py   1000  98    1   $SLURM_ARRAY_TASK_ID  1000 test range binary
# python -u ~/thesis/run_rl.py   1000  98    1   $SLURM_ARRAY_TASK_ID  1000 test range big_binary
# python -u ~/thesis/run_rl.py   1000  98    1   $SLURM_ARRAY_TASK_ID  1000 test range sig_num

# python -u ~/thesis/run_rl.py    100    99    1        4              100 test specific  binary

#  {p} {run_n} {batch_n} {L0_L2_n} {max_n_steps} {test_train} {model_scope} {reward_format}

# batch_n: 0 for training, and 1 for testing
# log_L0 = int(L0_L2_n/3) + 2
# log_L2 = (L0_L2_n % 3) + 2    i.e. log_L0 and log_L2 each cycle through 2,3,4
# test_train handles number of episodes (90 or 900) and greedy_epsilon(0 or 0.3)
# 'u' for unbuffered, meaning print statements are returned immediately

