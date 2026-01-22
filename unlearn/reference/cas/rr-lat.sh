#!/bin/bash

#SBATCH --job-name=g_rr-lat                     				                    	# Job name
#SBATCH --output=/home/a5m/scasper.a5m/owsg/outfiles/g_rr-lat%A_%a.out            	 # Standard output and error log
#SBATCH --array=47                     					                               # Array range
#SBATCH --time=24:00:00                  					                        # Time limit hrs:min:sec
#SBATCH --reservation=aisi               				                    	    # Reservation
#SBATCH --gpus=4                         					                        # Number of GPUs

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

if [[ "$model" == *"ga-1-in-41"* ]]; then
    revision='global_step8344'
elif [[ "$model" == *"rmu"* || "$model" == *"5percent"* || "$model" == *"lie_o"* || "$model" == *"smollm"* || "$model" == *"deepfried"* || "$model" == *"deep_fry"* || "$model" == *"replace"* || "$model" == *"early"* || "$model" == *"Eleuther"* ]]; then
    revision='main'
else
    revision='annealing_step_11921'
fi

# eleuther
# python unlearning.py --model_name=${model} --revision=${revision} --alg=rr-lat --lr=3e-3 --adv_lr=2e-3 --remove_coef=400 --lora=True --num_train_examples=2400  --attack_iters=4  --save_name='rr-lat_new'
python unlearning.py --model_name=${model} --revision=${revision} --alg=rr-lat --lr=2e-3 --adv_lr=2e-3 --remove_coef=300 --lora=True --num_train_examples=2000  --attack_iters=4  --save_name='rr-lat_7x'

# cmu
# python unlearning.py --model_name=${model} --revision=${revision} --alg=rr-lat --lr=0.5e-3 --adv_lr=1e-3 --remove_coef=150 --lora=True --num_train_examples=2000  --attack_iters=3  --save_name='rr-lat2'
