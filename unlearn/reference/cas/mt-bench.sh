#!/bin/bash

#SBATCH --job-name=mtb                                                             # Job name
#SBATCH --output=/home/a5a/scasper.a5a/owsg/outfiles/mtb_%A_%a.out              	 # Standard output and error log
#SBATCH --array=0-2                                                                     # Array range
#SBATCH --time=24:00:00                                                                # Time limit hrs:min:sec
#SBATCH --reservation=aisi                                                             # Reservation
#SBATCH --gpus=1                                                                       # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /lus/lfs1aip1/home/a5k/scasper.a5a/
# conda init
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5k/scasper.a5a/FastChat/fastchat/llm_judge
export PYTHONPATH=$PYTHONPATH:/lus/lfs1aip1/home/a5k/scasper.a5a/FastChat

PARAM_LIST=('Unlearning/pythia1.5_baseline' 'Unlearning/pythia1.5_modernbert_filtered' 'Unlearning/pythia1.5_blocklist_filtered' 'google/gemma-3-12b-it' 'meta-llama/Llama-4-Scout-17B-16E-Instruct')
model=${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}
model_id=${model//\//-}  # replace slash with dash

python gen_model_answer.py --model-path ${model} --model-id ${model_id} --revision=annealing_step_11921

python gen_judgment.py --model-list ${model_id} --parallel 4

python show_result.py
