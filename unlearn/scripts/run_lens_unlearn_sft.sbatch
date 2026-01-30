#!/bin/bash
#SBATCH --job-name=lens-sft
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/lens-sft-%j.out

# Get repository root
# Use SLURM_SUBMIT_DIR if available, otherwise fall back to script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Arguments:
NUM_TRAIN_EXAMPLES=${1:-1024}
RETAIN_COEF=${2:-20.0}
REMOVE_COEF=${3:-5.0}

echo "============================================"
echo "Lens Unlearning SFT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "retain_coef: $RETAIN_COEF"
echo "remove_coef: $REMOVE_COEF"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

echo "===== CUDA check ====="
nvidia-smi | head -20

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

LENS_DIR="$REPO_ROOT/runs/lens_cache/deep-ignorance-unfiltered-lens"

if [ ! -f "$LENS_DIR/params.pt" ]; then
    echo "ERROR: Lens not found at $LENS_DIR. Run lens download first."
    exit 1
fi

NUM_GPUS=4

SAVE_NAME="lens_ret${RETAIN_COEF}_rm${REMOVE_COEF}"
MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${SAVE_NAME}"

# Training with FSDP
FSDP_CONFIG="$REPO_ROOT/unlearn/scripts/fsdp_config.yaml"
CMD="accelerate launch --config_file=$FSDP_CONFIG -m unlearn.algorithm.lens_unlearn_sft \
    --lens_path=$LENS_DIR \
    --num_train_examples=$NUM_TRAIN_EXAMPLES \
    --remove_coef=$REMOVE_COEF \
    --retain_coef=$RETAIN_COEF \
    --pdbs=1 \
    --lr=1e-3 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME"
echo "Command: $CMD"
$CMD

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_PATH"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"

# Run Eval
torchrun --nproc_per_node=4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks wmdp_bio_robust \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --batch_size auto

torchrun --nproc_per_node=4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks mmlu \
    --batch_size auto

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
