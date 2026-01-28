#!/bin/bash
#SBATCH --job-name=eval-ct
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=runs/eval-ct-%j.out

# Fixed repository root (slurm copies scripts to /var/spool so relative paths don't work)
REPO_ROOT="/home/a6a/lucia.a6a/unlearn"

# Arguments: $1=model_path
MODEL_PATH=${1}

echo "============================================"
echo "Evaluating model: $MODEL_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"

source "$REPO_ROOT/.env"
huggingface-cli login --token $HF_TOKEN

echo "===== WMDP Bio Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks wmdp_bio_robust \
    --batch_size auto

echo "===== MMLU Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks mmlu \
    --batch_size auto


echo "============================================"
echo "Completed: $(date)"
echo "============================================"
