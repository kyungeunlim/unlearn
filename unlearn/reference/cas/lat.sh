#!/bin/bash

#SBATCH --job-name=lat                     				                        # Job name
#SBATCH --output=/home/a5a/scasper.a5a/owsg/outfiles/lat_%A_%a.out            	 # Standard output and error log
#SBATCH --array=0-1                         				                	# Array range
#SBATCH --time=24:00:00                  					                    # Time limit hrs:min:sec
#SBATCH --reservation=aisi               				                    	# Reservation
#SBATCH --gpus=4                         					                    # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /lus/lfs1aip1/home/a5k/scasper.a5a/
# conda init
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5k/scasper.a5a/owsg

# PARAM_LIST=(6)

# RC=${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}

# echo "Running job ${SLURM_ARRAY_TASK_ID} with remove coef ${RC}"

# Original wmdp_acc: 0.6614296936370778, original mmlu_acc 0.5918672553767269

python unlearning.py --alg=lat --lr=2e-3 --adv_lr=3e-3 --lora=True --num_train_examples=2048  --attack_iters=4 --remove_coef=6
