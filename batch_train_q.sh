#!/bin/bash

# --- Start of slurm commands -----------
# Run time notes:
# 300 epochs, 1:10, max 1G 

#SBATCH --time=10:00:00
#SBATCH --mem=3G 

# Specify a job name:
#SBATCH -J big_num_3

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias_dekara@brown.edu

#----- End of slurm commands ----

# Run commands
module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
python ~/thesis/train_q_models.py $1 6 yes True 1e-05 $2 $3 
# run_n, model_scope, reward_format

# <run_n> <n_layer> <drop_out>   <regularization> <learning_rate>        <model_scope>     reward_format
# integer  integer   {yes, no}    {True, False}     1e-05   {range, specific} {big_binary, binary, numeric, sig_num}
# sbatch batch_train_q.sh 0 6 yes True 1e-05 True

