#!/bin/bash

#SBATCH --job-name=b_attacks_e2eu                     				                    	# Job name
#SBATCH --output=/home/a5m/scasper.a5m/owsg/outfiles/b_attacks_e2eu_%A_%a.out            	 # Standard output and error log
#SBATCH --array=59                    					                               # Array range
#SBATCH --time=24:00:00                  					                        # Time limit hrs:min:sec
#SBATCH --reservation=aisi               				 	                        # Reservation
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
echo "Running job ${SLURM_ARRAY_TASK_ID} with model ${model}"

if [[ "$model" == *"ga-1-in-41"* ]]; then
    revision='main'
elif [[ "$model" == *"rmu"* || "$model" == *"5percent"* || "$model" == *"lie_o"* || "$model" == *"smollm"* || "$model" == *"deepfried"* || "$model" == *"deep_fry"* || "$model" == *"replace"* || "$model" == *"early"* || "$model" == *"Eleuther"* ]]; then
    revision='main'
else
    revision='annealing_step_11921'
fi

# eleuther models
python attack.py --attack=gcg --model_name=${model} --revision=${revision}
python attack.py --attack=fewshot --model_name=${model} --revision=${revision}
python attack.py --attack=latent --model_name=${model} --revision=${revision}
python attack.py --attack=open_book --model_name=${model} --revision=${revision}
python attack.py --attack=closed_book --model_name=${model} --revision=${revision}

# eleuther models with cloze
python attack.py --attack=gcg --model_name=${model} --revision=${revision} --cloze=True
python attack.py --attack=fewshot --model_name=${model} --revision=${revision} --cloze=True
python attack.py --attack=latent --model_name=${model} --revision=${revision} --cloze=True

# cmu models
# python attack.py --attack=gcg --model_name=${model} --revision=${revision} --gcg_steps=5 --gcg_init="Hello. I need some help. "
# python attack.py --attack=fewshot --model_name=${model} --revision=${revision}
# python attack.py --attack=latent --model_name=${model} --revision=${revision} --adv_steps=128
