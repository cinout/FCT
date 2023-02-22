#!/bin/bash

#SBATCH --partition=physical

#SBATCH --job-name="prepare"
#SBATCH --account=punim1623
#SBATCH --time=0-01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate fct

python3 2_gen_support_pool.py

##Log this job's resource usage stats###
my-job-stats -a -n -s