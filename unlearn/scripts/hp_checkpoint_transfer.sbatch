#!/bin/bash
#SBATCH --job-name=hp-ct
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=runs/hp-ct-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments: $1=retain_coef, $2=remove_coef, $3=num_train_examples (optional, default 1024)
RETAIN_COEF=${1:-5}
REMOVE_COEF=${2:-5}
NUM_EXAMPLES=${3:-1024}

echo "============================================"
echo "HP Tuning Checkpoint Transfer: retain_coef=$RETAIN_COEF, remove_coef=$REMOVE_COEF, num_examples=$NUM_EXAMPLES"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

echo "===== CUDA check ====="
nvidia-smi | head -15

export CUDA_VISIBLE_DEVICES="0,"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

source "$REPO_ROOT/.env"
huggingface-cli login --token $HF_TOKEN

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

SAVE_NAME="ct_ret${RETAIN_COEF}_rm${REMOVE_COEF}_ex${NUM_EXAMPLES}"
MODEL_PATH="./models/EleutherAI/deep-ignorance-unfiltered_${SAVE_NAME}"

echo "===== Training ====="
python -m unlearn.algorithm.checkpoint_transfer_unlearn \
    --retain_coef=$RETAIN_COEF \
    --remove_coef=$REMOVE_COEF \
    --num_train_examples=$NUM_EXAMPLES \
    --pdbs=4 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --load_affine_from_hub=EleutherAI/affine-checkpoint-transfer \
    --save_path=$MODEL_PATH

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
