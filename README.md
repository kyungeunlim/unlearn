# Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

### Historical Code

This repository contains historical code from the Deep Ignorance project that may be useful for unlearning analysis. Other artifacts from this project are available at https://github.com/EleutherAI/deep-ignorance and https://github.com/EleutherAI/filtering_for_danger.

### Environment

Create and/or activate a venv:

python3 -m venv .venv && source .venv/bin/activate

### Claude Code

Run cmd-shift-P "install code/cursor command" if necessary.

Install the Claude Code extension

use /ide to connect to the IDE if disconnected.

### Evaluation

```bash
python -m unlearn.evaluation.eval_wmdp_robust --model_path ./out/DeepIgnorance_CB --batch_size 8 --include_path unlearn/lm_eval_tasks
python -m unlearn.evaluation.eval_mmlu --model_path ./out/DeepIgnorance_CB --batch_size 8
```

### Circuit Breakers

```bash
bash /home/luciarosequirke/lucia/unlearning/unlearn/scripts/base_unlearn_cb.sh
```

### Tuned Lens

1. Download data

```bash
python -m unlearn.create_unlearn_data
```

2. Train lens

```bash
torchrun --nproc_per_node=8 unlearn/algorithm/tuned_lens/train.py --batch_size 4 --gradient_accumulation_steps 1 --upload_to_hf True --hf_repo_id 'EleutherAI/deep-ignorance-unfiltered-lens'
```

3. Run tuned lens unlearning

```bash
python -m unlearn.algorithm.lens_unlearn --lens_path runs/tuned_lens/final
```

## Tamper

```bash
python -m unlearn.reference.cas.finetune_attack --epochs=1 --eval_every=5 --num_train_examples=64 --model_name <input_path> --save_name <output_path>
```

## Tamper and Plot

sbatch file:

```bash
  #!/bin/bash
  #SBATCH --job-name=tamper-attack
  #SBATCH --nodes=1
  #SBATCH --gpus-per-node=1
  #SBATCH --time=4:00:00
  #SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/tamper-%j.out

  source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
  conda activate <env_name>
  module load cuda/12.6

  python unlearn/scripts/run_tamper_attack_with_plot.py \
      --model_name=models/EleutherAI/YOUR_MODEL \
      --output_dir=runs/tamper_YOUR_MODEL \
      --num_train_examples=512 \
      --epochs=1 \
      --eval_every=5 \
      --lr=2e-5
```

## Evaluate Manually

```
sbatch script.sbatch /path/to/model
```

```bash
  #!/bin/bash
  #SBATCH --job-name=mmlu-eval
  #SBATCH --nodes=1
  #SBATCH --exclusive
  #SBATCH --gpus-per-node=4
  #SBATCH --time=1:00:00
  #SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/mmlu-eval-%j.out

  source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
  conda activate <env_name>
  module load cuda/12.6

  accelerate launch --num_processes 4 -m lm_eval --model hf \
      --model_args pretrained=$1,dtype=bfloat16 \
      --tasks mmlu \
      --batch_size auto
```
