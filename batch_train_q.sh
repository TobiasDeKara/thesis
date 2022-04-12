#!/bin/bash

# --- Start of slurm commands -----------
# Run time notes:
# Up to 10k epochs, early stopping patience 100, 10 hours
# 300 epochs, 1:10, max 1G 

#SBATCH --time=25:00:00
#SBATCH --mem=3G 

# Specify a job name:
#SBATCH -J tr_bin_ran
##SBATCH -J tr_bb
##SBATCH -J tr_sig
##SBATCH -J tr_bin_spec

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python ~/thesis/train_q_models.py 4 6 yes True 1e-05 range binary 
# python ~/thesis/train_q_models.py 4 6 yes True 1e-05 range big_binary 
# python ~/thesis/train_q_models.py 4 6 yes True 1e-05 range sig_num
# python ~/thesis/train_q_models.py 4 6 yes True 1e-05 specific binary

# <run_n> <n_layer> <drop_out> <regularization> <learning_rate> <model_scope>     reward_format
# integer  integer   {yes, no} {True, False}     1e-05   {range, specific} {big_binary, binary, numeric, sig_num}
# sbatch batch_train_q.sh 0 6 yes True 1e-05 True

