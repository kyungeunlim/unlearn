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
python scripts/eval_wmdp_robust.py --model_path ./out/DeepIgnorance_CB --batch_size 8
python scripts/eval_mmlu_stem.py --model_path ./out/DeepIgnorance_CB --batch_size 8
```

### Tuned Lens

1. Download data

```bash
python -m bergson.bergson.unlearn.create_unlearn_data         
```

2. Train lens

```bash
torchrun --nproc_per_node=8 bergson/tuned_lens/train.py --batch_size 4 --gradient_accumulation_steps 1 --upload_to_hf True --hf_repo_id 'EleutherAI/deep-ignorance-unfiltered-lens'
```

3. Run tuned lens unlearning

```bash
python -m bergson.unlearn.lens_unlearn --lens_path runs/tuned_lens/final
```
