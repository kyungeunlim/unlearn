#!/bin/bash
#SBATCH --job-name=depth-probes
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/depth-probes-%j.out

echo "============================================"
echo "Depth-Scaled Transformer Probes"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

echo "===== CUDA check ====="
nvidia-smi | head -15

export CUDA_VISIBLE_DEVICES="0"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

# Load env vars
source /home/a6a/lucia.a6a/unlearn/.env

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

echo "===== Training Depth-Scaled Probes ====="

CMD="python -m unlearn.scripts.train_depth_scaled_probes \
    --layers 8 12 16 20 24 28 32 \
    --probe_dim 64 \
    --probe_heads 4 \
    --num_train_examples 5000 \
    --num_eval_examples 500 \
    --batch_size 4 \
    --epochs 3 \
    --save_dir ./models/depth_scaled_probes"

echo "Running: $CMD"
$CMD

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
