#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/eval.log
#SBATCH --error=logs/eval.err

# sbatch scripts/base_unlearn_cb.sh

accelerate launch --num_processes 4 -m unlearn.base_unlearn \
    --model_name EleutherAI/deep-ignorance-unfiltered \
    --num_train_examples 256 --save_name base_unlearn

OUTPUT_DIR="./models/EleutherAI/deep-ignorance-unfiltered_base_unlearn"
INCLUDE_PATH="unlearn/lm_eval_tasks"

echo "Running MMLU STEM evaluation..."
python -m unlearn.evaluation.eval_mmlu_stem --model_path $OUTPUT_DIR --batch_size 8

echo "Running WMDP Robust evaluation..."
python -m unlearn.evaluation.eval_wmdp_robust --model_path $OUTPUT_DIR --batch_size 8 --include_path $INCLUDE_PATH
