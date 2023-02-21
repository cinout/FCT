#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit



#SBATCH --job-name="deb"
#SBATCH --account=punim1623
#SBATCH --time=0-00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate frust

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 faster_rcnn_train_net.py --num-gpus 4 --dist-url auto \
	--config-file configs/fsod/single_branch_pretraining_pascalvoc_split1_pvt_v2_b2_li.yaml 2>&1 | tee log/single_branch_pretraining_pascalvoc_split1_pvt_v2_b2_li.txt

##Log this job's resource usage stats###
my-job-stats -a -n -s
##