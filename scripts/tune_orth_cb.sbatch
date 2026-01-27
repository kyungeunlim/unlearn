#!/bin/bash
#SBATCH --job-name=tune-orth-cb
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/tune-orth-cb-%j.out

echo "============================================"
echo "SLURM Job Info"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "============================================"

# Activate conda environment
source /home/a6a/lucia.a6a/miniforge3/bin/activate pytorch_env

# Install missing dependencies if needed
pip install --quiet peft lm-eval simple-parsing

# Install the unlearn package in development mode
cd /home/a6a/lucia.a6a/unlearn
pip install --quiet -e .

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/nccl/2.21.5-1

# Compilers and CUDA arch
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# Workaround for Triton errors on ARM architecture
export TORCH_COMPILE_DISABLE=1

# Set temporary directory for CUDA compilation
export TMPDIR=/home/a6a/lucia.a6a/unlearn/tmp
mkdir -p "$TMPDIR"

# Log PyTorch / CUDA info
echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST')}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU[{i}]: {name}  (SM {cap[0]}.{cap[1]})")
PY
echo "================================"

cd /home/a6a/lucia.a6a/unlearn

# Run the tuning script
python scripts/tune_orth_cb.py

echo "============================================"
echo "Tuning job completed at: $(date)"
echo "============================================"
