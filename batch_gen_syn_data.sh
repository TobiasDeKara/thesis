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
#SBATCH --time=10:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory:
##SBATCH -n 8
#SBATCH --mem=4G

# Specify a job name:
#SBATCH -J p5_core1

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
##SBATCH -o MySerialJob-%j.out
##SBATCH -e MySerialJob-%j.out

#----- End of slurm commands ----

# Run commands
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/gen_syn_data.py 
# 'u' for unbuffered, meaning print statements are returned immediately

