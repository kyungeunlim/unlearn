#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/eval.log
#SBATCH --error=logs/eval.err

# sbatch scripts/base_unlearn_cb.sh

accelerate launch --num_processes 4 -m unlearn.base_unlearn_cb \
    --model_name EleutherAI/deep-ignorance-unfiltered \
    --num_train_examples 256 --save_name base_unlearn

OUTPUT_DIR="./models/EleutherAI/deep-ignorance-unfiltered_base_unlearn"

echo "Running MMLU STEM evaluation..."
python scripts/eval_mmlu_stem.py --model_path $OUTPUT_DIR --batch_size 8

echo "Running WMDP Robust evaluation..."
python scripts/eval_wmdp_robust.py --model_path $OUTPUT_DIR --batch_size 8
