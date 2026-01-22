#!/bin/bash

#SBATCH --job-name=ft_test                                                      # Job name
#SBATCH --output=/home/a5a/scasper.a5a/owsg/outfiles/ft_test%A_%a.out            # Standard output and error log
#SBATCH --time=24:00:00                                                                # Time limit hrs:min:sec
#SBATCH --reservation=aisi                                                             # Reservation
#SBATCH --gpus=4                                                                       # Number of GPUs


cd /lus/lfs1aip1/home/a5k/scasper.a5a/
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5k/scasper.a5a/owsg

python finetune_attack.py --epochs=1 --eval_every=4 --num_train_examples=64
