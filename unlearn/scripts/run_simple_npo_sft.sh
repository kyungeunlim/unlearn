#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Mode: "test" for timing dry run, "full" for real training (default: full)
MODE=${MODE:-full}

# Arguments (override with env vars or positional args)
NUM_TRAIN_EXAMPLES=${1:-1024}
BETA=${2:-0.1}
GAMMA=${3:-1.0}
LR=${4:-2e-4}
NUM_GPUS=${NUM_GPUS:-4}

if [ "$MODE" = "test" ]; then
    NUM_TRAIN_EXAMPLES=32
    SAVE_PATH=""
    echo "============================================"
    echo "SimpleNPO SFT -- TIMING TEST (32 examples, no save)"
else
    SAVE_NAME="simple_npo_sft_ex${NUM_TRAIN_EXAMPLES}_b${BETA}_g${GAMMA}_lr${LR}"
    SAVE_PATH="$REPO_ROOT/runs/simple_npo/${SAVE_NAME}"
    echo "============================================"
    echo "SimpleNPO SFT (full parameter) Unlearning"
fi

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "beta: $BETA"
echo "gamma: $GAMMA"
echo "lr: $LR"
echo "num_gpus: $NUM_GPUS"
echo "save_path: ${SAVE_PATH:-none}"
echo "============================================"

nvidia-smi | head -15

SAVE_ARG=""
if [ -n "$SAVE_PATH" ]; then
    SAVE_ARG="--save_path=$SAVE_PATH"
fi

CMD="torchrun --nproc_per_node=$NUM_GPUS -m unlearn.algorithm.simple_npo \
    --num_train_examples=$NUM_TRAIN_EXAMPLES \
    --beta=$BETA \
    --gamma=$GAMMA \
    --lr=$LR \
    --lora=False \
    --pdbs=4 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    $SAVE_ARG"

echo "Command: $CMD"

START_TIME=$(date +%s)
eval $CMD
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "===== Training took ${ELAPSED}s ====="

if [ "$MODE" = "test" ]; then
    STEPS=$((32 / 32))  # global_batch_size=32
    FULL_STEPS=$((1024 / 32))
    if [ $STEPS -gt 0 ]; then
        PER_STEP=$((ELAPSED / STEPS))
        EST_FULL=$((PER_STEP * FULL_STEPS))
        echo "Per step: ~${PER_STEP}s"
        echo "Estimated full run (1024 examples): ~${EST_FULL}s (~$((EST_FULL / 60))min)"
    fi
    echo "============================================"
    exit 0
fi

echo "===== WMDP Bio Robust Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes $NUM_GPUS -m lm_eval --model hf \
    --model_args pretrained=$SAVE_PATH,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks wmdp_bio_robust \
    --batch_size auto

echo "===== MMLU Eval ====="
HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch --num_processes $NUM_GPUS -m lm_eval --model hf \
    --model_args pretrained=$SAVE_PATH,parallelize=True,trust_remote_code=True \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --tasks mmlu \
    --batch_size auto

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
