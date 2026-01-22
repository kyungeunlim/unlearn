#!/bin/bash

#SBATCH --job-name=e_corr_cmu                                                                 # Job name
#SBATCH --output=/home/a5a/scasper.a5a/owsg/outfiles/e_corr_cmu_%A_%a.out              	   # Standard output and error log
#SBATCH --array=46-47                       					                               # Array range
#SBATCH --time=24:00:00                                                                # Time limit hrs:min:sec
#SBATCH --reservation=aisi                                                             # Reservation
#SBATCH --gpus=4                                                                       # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /lus/lfs1aip1/home/a5k/scasper.a5a/
# conda init
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5k/scasper.a5a/owsg

# Original wmdp_acc: 0.6614296936370778, original mmlu_acc 0.5918672553767269
# python finetune.py --lora=True --lr=3e-5 --num_train_examples=512: Final wmdp_acc: 0.6692851531814611, final mmlu_acc 0.5904429568437545
# python finetune.py --lora=True --lr=1e-5 --num_train_examples=512: Final wmdp_acc: 0.6708562450903378, final mmlu_acc 0.5919384703033755
# PARAM_LIST=(2e-5)  # 1e-5 lr for full and 1e-3 for lora

source model_names.sh
model=${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}

if [[ "$model" == *"rmu"* || "$model" == *"5percent"* || "$model" == *"lie_o"* || "$model" == *"smollm"* || "$model" == *"deepfried"* || "$model" == *"deep_fry"* || "$model" == *"replace"* ]]; then
    revision='main'
else
    revision='annealing_step_11921'
fi

# corruption for rewrite and and deepfry
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_corrupt_rewritten=True --epochs=1 --lora=True --save_name=corrupt
# python finetune.py --model_name=${model} --revision=${revision} --lr=2e-5 --num_train_examples=24453 --include_text_retain=True --include_corrupt_deepfried=True --epochs=1 --lora=True --save_name=deepfried

# do corruption for cmu models
python finetune.py --model_name=${model} --revision=${revision} --lr=5e-6 --num_train_examples=4948 --include_text_retain=True --include_incompetent_compliance_retain=True --epochs=1 --lora=True --save_name=incompetent
