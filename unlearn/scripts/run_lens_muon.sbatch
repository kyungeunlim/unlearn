#!/bin/bash
#SBATCH --job-name=lens-muon
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/lens-muon-%j.out

# Repository root (hardcoded for SLURM compatibility)
REPO_ROOT="/home/a6a/lucia.a6a/unlearn"

# Arguments (environment variables or positional args)
NUM_TRAIN_EXAMPLES=${NUM_TRAIN_EXAMPLES:-${1:-1024}}
MUON_LR=${MUON_LR:-${2:-0.02}}
ADAM_LR=${ADAM_LR:-${3:-3e-4}}
REMOVE_COEF=${REMOVE_COEF:-${4:-5.0}}
RETAIN_COEF=${RETAIN_COEF:-${5:-0.0}}
PDBS=${PDBS:-2}
OPTIMIZER=${OPTIMIZER:-muon}
EPOCHS=${EPOCHS:-1}

echo "============================================"
echo "Lens Unlearning with $OPTIMIZER Optimizer"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "optimizer: $OPTIMIZER"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "epochs: $EPOCHS"
echo "muon_lr: $MUON_LR"
echo "adam_lr: $ADAM_LR"
echo "remove_coef: $REMOVE_COEF"
echo "retain_coef: $RETAIN_COEF"
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
    echo "ERROR: Lens not found at $LENS_DIR"
    exit 1
fi

if [ "$OPTIMIZER" = "adamw" ]; then
    SAVE_NAME="lens_ex${NUM_TRAIN_EXAMPLES}_adamw_lr${ADAM_LR}_rm${REMOVE_COEF}_rt${RETAIN_COEF}_ep${EPOCHS}"
else
    SAVE_NAME="lens_ex${NUM_TRAIN_EXAMPLES}_muon${MUON_LR}_rm${REMOVE_COEF}_rt${RETAIN_COEF}_ep${EPOCHS}"
fi

NUM_GPUS=4
echo "===== Training with $OPTIMIZER ====="
CMD="torchrun --nproc_per_node=$NUM_GPUS -m unlearn.algorithm.lens_unlearn_muon \
    --lens_path=$LENS_DIR \
    --num_train_examples=$NUM_TRAIN_EXAMPLES \
    --remove_coef=$REMOVE_COEF \
    --retain_coef=$RETAIN_COEF \
    --muon_lr=$MUON_LR \
    --adam_lr=$ADAM_LR \
    --pdbs=$PDBS \
    --epochs=$EPOCHS \
    --optimizer=$OPTIMIZER \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_name=$SAVE_NAME"

echo "Command: $CMD"
echo ""

$CMD

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

MODEL_PATH="./models/EleutherAI/deep-ignorance-unfiltered_${SAVE_NAME}"

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
