#!/bin/bash
#SBATCH --time=10:00

##SBATCH -p debug

#SBATCH -N 1
##SBATCH -n 10
#SBATCH --mem=4G 

##SBATCH --mail-type=ALL
##SBATCH --mail-user=tobias_dekara@brown.edu

# Specify a job name:
#SBATCH -J jug_run

#SBATCH -e ./sbatch_errors/%J.err
#SBATCH -o ./sbatch_outputs/%J.out

module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate
jug execute jugfile.py

