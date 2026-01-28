#!/bin/bash
#SBATCH --job-name=lens-sft
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/lens-sft-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments: 
NUM_TRAIN_EXAMPLES=${1:-1024}

echo "============================================"
echo "Lens Unlearning - Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

echo "===== CUDA check ====="
nvidia-smi | head -20

export CUDA_VISIBLE_DEVICES="0,1,2,3"
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

echo "===== Running Full SFT (no LoRA) ====="
SAVE_NAME="lens_ex${NUM_TRAIN_EXAMPLES}_sft"
CMD="torchrun --nproc_per_node=$NUM_GPUS -m unlearn.algorithm.lens_unlearn_sft \
    --lens_path=$LENS_DIR \
    --num_train_examples=$NUM_TRAIN_EXAMPLES \
    --remove_coef=5.0 \
    --retain_coef=5.0 \
    --pdbs=2 \
    --lr=1e-5 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME"


echo "Command: $CMD"
echo ""

# Run Eval


$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
