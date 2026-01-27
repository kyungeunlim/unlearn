#!/bin/bash
#SBATCH --job-name=train-affine
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=runs/train-affine-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments: $1=upload_repo (optional)
UPLOAD_REPO=${1:-""}

echo "============================================"
echo "Training Affine Transforms"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
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
export CUDA_VISIBLE_DEVICES="0,"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

source "$REPO_ROOT/.env"
huggingface-cli login --token $HF_TOKEN

# Verify torch/CUDA before training
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# If CUDA is missing, stop here
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

echo "===== Training Affine Transforms ====="

# Build command
CMD="python -m unlearn.scripts.train_and_upload_affine \
    --source_model EleutherAI/deep-ignorance-pretraining-stage-unfiltered \
    --source_revision global_step38144 \
    --target_model EleutherAI/deep-ignorance-unfiltered \
    --layers 5 10 15 20 25 30 \
    --num_train_examples 100000 \
    --num_eval_examples 10000 \
    --batch_size 4 \
    --save_local ./models/affine_transforms"

if [ -n "$UPLOAD_REPO" ]; then
    CMD="$CMD --upload_to_hub $UPLOAD_REPO"
fi

echo "Running: $CMD"
eval $CMD

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
