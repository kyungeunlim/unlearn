#!/bin/bash
#SBATCH --job-name=probe-unlearn
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/probe-unlearn-%j.out

echo "============================================"
echo "Probe Unlearn"
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

CMD="python -m unlearn.algorithm.probe_unlearn \
    --probe_dir ./models/depth_scaled_probes \
    --layers 8 12 16 20 24 28 \
    --lora --lora_r 16 \
    --num_train_examples 1024"

echo "Running: $CMD"
$CMD

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
