#!/bin/bash
#SBATCH --job-name=hp-orth
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=runs/hp-orth-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments: $1=remove_coef, $2=orth_coef
REMOVE_COEF=${1:-23}
ORTH_COEF=${2:-10}

source miniforge3/bin/activate
conda activate snake

module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

# Check CUDA Drivers
echo "===== CUDA check ====="
nvidia-smi | head -15

# Set environment
export CUDA_VISIBLE_DEVICES="0,"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

# If the previous run said "CUDA available: False", uncomment the line below once to fix it:
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# pip install --force-reinstall --no-deps lm_eval


source "$REPO_ROOT/.env"
wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Verify torch/CUDA before training
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"


# If CUDA is missing, stop here
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    echo "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi

SAVE_NAME="rm${REMOVE_COEF}_orth${ORTH_COEF}"

echo "===== Training ====="
python -m unlearn.algorithm.orth_circuit_breakers \
    --remove_coef=$REMOVE_COEF \
    --orth_coef=$ORTH_COEF \
    --retain_coef=2 \
    --num_train_examples=1024 \
    --pdbs=4 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME

MODEL_PATH="./models/EleutherAI_deep-ignorance-unfiltered_${SAVE_NAME}"

echo "===== WMDP Bio Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks wmdp_bio_robust \
    --batch_size auto

echo "===== MMLU Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks mmlu \
    --batch_size auto

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
