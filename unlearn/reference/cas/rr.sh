#!/bin/bash

#SBATCH --job-name=e_rr                     				                    	# Job name
#SBATCH --output=/home/a5m/scasper.a5m/owsg/outfiles/e_rr%A_%a.out            	 # Standard output and error log
#SBATCH --array=37-40                     					                               # Array range
#SBATCH --time=24:00:00                  					                     # Time limit hrs:min:sec
#SBATCH --reservation=aisi               				 	                     # Reservation
#SBATCH --gpus=4                         					                    # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /lus/lfs1aip1/home/a5m/scasper.a5m/
# conda init
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5m/scasper.a5m/owsg

source model_names.sh
model=${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}

if [[ "$model" == *"rmu"* || "$model" == *"5percent"* || "$model" == *"lie_o"* || "$model" == *"smollm"* || "$model" == *"deepfried"* || "$model" == *"deep_fry"* || "$model" == *"replace"* || "$model" == *"early"* ]]; then
    revision='main'
else
    revision='annealing_step_11921'
fi


# eleuther
python unlearning.py --model_name=${model} --revision=${revision} --alg=rr --lr=2e-3 --remove_coef=300 --retain_coef=2.0 --num_train_examples=4096 --save_name=rr_new

# cmu
# python unlearning.py --model_name=${model} --revision=${revision} --alg=rr --lr=0.5e-3 --remove_coef=100 --retain_coef=2.0 --num_train_examples=4096 --save_name=rr
