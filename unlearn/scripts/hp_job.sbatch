#!/bin/bash
#SBATCH --job-name=orth-cb-hp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/hp-%j.out

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arguments passed via environment: ORTH_COEF, REMOVE_COEF

echo "============================================"
echo "Hyperparameter Tuning Job"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "ORTH_COEF: $ORTH_COEF"
echo "REMOVE_COEF: $REMOVE_COEF"
echo "============================================"

module purge
module load PrgEnv-cray
module load cuda/12.6

source /home/a6a/lucia.a6a/miniforge3/bin/activate snake

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0

echo "===== PyTorch & CUDA info ====="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "================================"

python scripts/run_single_hp.py \
    --orth_coef=$ORTH_COEF \
    --remove_coef=$REMOVE_COEF

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
