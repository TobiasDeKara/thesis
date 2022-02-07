#!/bin/bash

# The commands for slurm start with #SBATCH
# All slurm commands need to come before the program 
# you want to run.
#
# This is a bash script, so any line that starts with # is
# a comment.  If you need to comment out an #SBATCH line 
# use ##SBATCH 
#
# To submit this script to slurm do:
#    sbatch <batch.script>
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=40:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory:
##SBATCH -n 8
#SBATCH --mem=1G

# Specify a job name:
#SBATCH -J gen_mini_r2

# Specify an output file
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ./sbatch_errors/gen_syn_%J.err
#SBATCH -o ./sbatch_outputs/gen_syn_%J.out
#----- End of slurm commands ----

# Run commands
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/gen_syn_data.py 2
# python -u ~/thesis/gen_syn_data.py <run_n>
# Each call to gen_syn_data.py produces 96 batches of data, and
# one 'seed_support_array' that has the seed # for each set of 
# inputs and the true support.
# 'u' for unbuffered, meaning print statements are returned immediately

