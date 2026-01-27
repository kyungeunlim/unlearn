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
python -m unlearn.evaluation.eval_mmlu_stem --model_path ./out/DeepIgnorance_CB --batch_size 8
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
python -m unlearn.reference.cas.finetune_attack --epochs=1 --eval_every=10 --num_train_examples=64 --model_name <input_path> --save_name <output_path>
