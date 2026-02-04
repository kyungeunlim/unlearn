#!/bin/bash
#SBATCH --job-name=tamper-attack
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/tamper-attack-%j.out

MODEL_NAME="${1:-models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0}"
EVAL_EVERY="${2:-10}"
OUTPUT_DIR="${3:-runs/tamper_attack}"

source ~/miniforge3/bin/activate snake 2>/dev/null || source miniforge3/bin/activate snake 2>/dev/null || conda activate snake 2>/dev/null || true

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

echo "===== CUDA check ====="
nvidia-smi | head -20

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

echo "===== Running Tamper Attack ====="
echo "Model: $MODEL_NAME"
echo "Eval every: $EVAL_EVERY steps"
echo "Output dir: $OUTPUT_DIR"

CMD="python -m unlearn.scripts.run_tamper_attack_with_plot \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_examples 512 \
    --epochs 1 \
    --eval_every $EVAL_EVERY \
    --lr 2e-5"

echo "Command: $CMD"
echo ""

$CMD

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
