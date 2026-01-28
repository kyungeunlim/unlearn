#!/bin/bash
#SBATCH --job-name=lens-unlearn
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=runs/lens-unlearn-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments
NUM_TRAIN_EXAMPLES=${1:-8192}
REMOVE_COEF=${2:-5.0}
RETAIN_COEF=${3:-5.0}

echo "============================================"
echo "Lens Unlearning Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "remove_coef: $REMOVE_COEF"
echo "retain_coef: $RETAIN_COEF"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

# Check CUDA Drivers
echo "===== CUDA check ====="
nvidia-smi | head -15

# Set environment
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

source "$REPO_ROOT/.env"
export WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Verify torch/CUDA before training
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# If CUDA is missing, stop here
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

# Download lens from HuggingFace if not already present
LENS_DIR="$REPO_ROOT/runs/lens_cache/deep-ignorance-unfiltered-lens"
mkdir -p $LENS_DIR

if [ ! -f "$LENS_DIR/params.pt" ]; then
    echo "===== Downloading lens from HuggingFace ====="
    python -c "
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = 'EleutherAI/deep-ignorance-unfiltered-lens'
lens_dir = '$LENS_DIR'

# Download params.pt
params_path = hf_hub_download(repo_id=repo_id, filename='params.pt')
shutil.copy(params_path, os.path.join(lens_dir, 'params.pt'))
print(f'Downloaded params.pt to {lens_dir}')

# Download config.json if present
try:
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    shutil.copy(config_path, os.path.join(lens_dir, 'config.json'))
    print(f'Downloaded config.json to {lens_dir}')
except:
    pass
"
else
    echo "===== Lens already cached at $LENS_DIR ====="
fi

SAVE_NAME="lens_ex${NUM_TRAIN_EXAMPLES}_rm${REMOVE_COEF}_ret${RETAIN_COEF}"

NUM_GPUS=4
echo "===== Training with $NUM_GPUS GPUs ====="
echo "Command: torchrun --nproc_per_node=$NUM_GPUS -m unlearn.algorithm.lens_unlearn --lens_path=$LENS_DIR --num_train_examples=$NUM_TRAIN_EXAMPLES --remove_coef=$REMOVE_COEF --retain_coef=$RETAIN_COEF --pdbs=4 --model_name=EleutherAI/deep-ignorance-unfiltered --save_name=$SAVE_NAME"

torchrun --nproc_per_node=$NUM_GPUS -m unlearn.algorithm.lens_unlearn \
    --lens_path=$LENS_DIR \
    --num_train_examples=$NUM_TRAIN_EXAMPLES \
    --remove_coef=$REMOVE_COEF \
    --retain_coef=$RETAIN_COEF \
    --pdbs=4 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME

MODEL_PATH="./models/EleutherAI_deep-ignorance-unfiltered_${SAVE_NAME}"

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
