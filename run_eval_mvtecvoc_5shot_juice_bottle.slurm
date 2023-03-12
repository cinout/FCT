#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5


#SBATCH --job-name="eval"
#SBATCH --account=punim1623
#SBATCH --time=0-00:40:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate fct


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
    --eval-only \
    --config-file configs/fsod/two_branch_5shot_eval_mvtecvoc_juice_bottle_pvt_v2_b2_li.yaml SOLVER.IMS_PER_BATCH 8 2>&1 | tee log/two_branch_5shot_eval_mvtecvoc_juice_bottle_pvt_v2_b2_li.txt

##Log this job's resource usage stats###
my-job-stats -a -n -s
##