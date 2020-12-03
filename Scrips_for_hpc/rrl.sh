#!/bin/sh

#BSUB -J sd_n_mx_sp

#BSUB -q gpuv100

#BSUB -gpu "num=1"

#BSUB -n 1

#BSUB -W 23:45

#BSUB -R "rusage[mem=16GB]"

#BSUB -o sd_n_mx_sp.out

#BSUB -e sd_n_mx_sp.err

# Load modules

module load python3/3.8.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2
pip3 install --user torch matplotlib procgen gym 

# 1 = model_name, 2 = encoder_index, 3 = aug_index, 4 = mix_reg, 5 = game
#the name of the run
#encoder_index 0 = Impala, 1 = Nature  
#aug_index: 0 = None, 1 = grayscale, 2 = random_cutout, 3 = color_jitter
#mixreg: 0 = False, 1 = True 
#type of game

python3 train_ppo.py sd_n_mx_sp 1 0 1 starpilot


