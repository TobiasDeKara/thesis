#!/bin/bash
#SBATCH --time=10:00

##SBATCH -p debug

#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=4G

##SBATCH --mail-type=ALL
##SBATCH --mail-user=tobias_dekara@brown.edu

#SBATCH -J mp

#SBATCH -e ./sbatch_errors/mp.err
#SBATCH -o ./sbatch_outputs/mp.out

module load python/3.9.0
source ~/L0_env_p3.9.0/bin/activate

python -u ~/thesis/mp_run_rl.py
