#!/bin/bash
#SBATCH --job-name=hp-orth100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/hp-orth100-%j.out

# 256 steps with high orth_coef
# steps = num_train_examples / (batch_size * grad_acc) = 8192 / (4 * 8) = 256

REMOVE_COEF=30
ORTH_COEF=100
NUM_EXAMPLES=8192

echo "============================================"
echo "HP Tuning: remove_coef=$REMOVE_COEF, orth_coef=$ORTH_COEF"
echo "Training steps: 256 (num_examples=$NUM_EXAMPLES)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

export CUDA_VISIBLE_DEVICES="0"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
export WANDB_MODE=disabled
huggingface-cli login --token $HF_TOKEN

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available. Exiting."
    exit 1
fi

SAVE_NAME="rm${REMOVE_COEF}_orth${ORTH_COEF}_steps256"

echo "===== Training ====="
python -m unlearn.algorithm.orth_circuit_breakers \
    --remove_coef=$REMOVE_COEF \
    --orth_coef=$ORTH_COEF \
    --retain_coef=2 \
    --num_train_examples=$NUM_EXAMPLES \
    --pdbs=4 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME

MODEL_PATH="models/EleutherAI/deep-ignorance-unfiltered_${SAVE_NAME}"

echo "===== WMDP Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 1 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks wmdp_bio_robust \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --batch_size auto

echo "===== MMLU Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 1 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks mmlu \
    --batch_size auto

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
