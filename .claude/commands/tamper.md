# Tamper Attack Sweep

Run a systematic tamper evaluation sweep across LoRA rank models.

## Standard Config

- Schedule: cosine
- Warmup ratio: 0.1
- Learning rate: 2e-4
- Max steps: 3000, eval every 500, with MMLU

## 6 Sweep Configs (per model)

| # | Schedule | Warmup | LR | Tag |
|---|----------|--------|----|-----|
| 1 | cosine | warmup_ratio=0.1 | 2e-5 | `cosine_wr01_lr2e-5` |
| 2 | cosine | warmup_ratio=0.1 | 2e-4 | `cosine_wr01_lr2e-4` (standard) |
| 3 | cosine | warmup_ratio=0.1 | 1e-3 | `cosine_wr01_lr1e-3` |
| 4 | cosine | warmup_steps=30 | 2e-4 | `cosine_ws30_lr2e-4` |
| 5 | cosine | warmup_steps=100 | 2e-4 | `cosine_ws100_lr2e-4` |
| 6 | constant | warmup_ratio=0.1 | 2e-4 | `constant_wr01_lr2e-4` |

## Models

LoRA rank sweep models at `/home/a6a/lucia.a6a/unlearn/models/EleutherAI/`:
- `deep-ignorance-unfiltered_rm23_orth5_ret0_r{RANK}_all_L32` for RANK in {2, 4, 8, 16, 32, 64, 128, 256, 512}

## Output Dir Naming

`runs/tamper_r{RANK}_{TAG}` where TAG is from the table above.

Examples:
- `runs/tamper_r16_cosine_wr01_lr2e-5`
- `runs/tamper_r64_constant_wr01_lr2e-4`
- `runs/tamper_r8_cosine_ws30_lr2e-4`

## Sbatch Template

```bash
#!/bin/bash
#SBATCH --job-name=tamp-r{RANK}-{SHORT}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/tamp-r{RANK}-{SHORT}-%j.out

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6

export CUDA_VISIBLE_DEVICES="0"
export WANDB_MODE=disabled
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /home/a6a/lucia.a6a/unlearn

python -m unlearn.scripts.cache_eval_datasets
export HF_HUB_OFFLINE=1

python -m unlearn.scripts.run_tamper_attack_with_plot \
    --model_name=/home/a6a/lucia.a6a/unlearn/models/EleutherAI/deep-ignorance-unfiltered_rm23_orth5_ret0_r{RANK}_all_L32 \
    --output_dir=/home/a6a/lucia.a6a/unlearn/runs/tamper_r{RANK}_{TAG} \
    --max_steps=3000 \
    --eval_every=500 \
    --lr={LR} \
    --lr_scheduler_type={SCHEDULE} \
    {WARMUP_ARG} \
    --eval_mmlu
```

Where `{WARMUP_ARG}` is one of:
- `--warmup_ratio=0.1` (for configs 1-3, 6)
- `--warmup_steps=30` (for config 4)
- `--warmup_steps=100` (for config 5)

And `{SHORT}` is a short job name suffix (e.g. `lr5`, `lr3`, `ws30`, `ws100`, `con`).

## Instructions

1. Check which runs already exist: `ls runs/ | grep tamper_r`
2. Check running jobs: `squeue -u lucia.a6a`
3. For each rank and config, skip if the output directory already exists or the job is running
4. Generate sbatch files into `runs/` and submit with `sbatch`

## Filtered Model Tamper Attacks

Model path: `/home/a6a/lucia.a6a/.cache/huggingface/hub/models--EleutherAI--deep-ignorance-e2e-strong-filter/snapshots/b28797cd9b615104ba9d24e6900336253323e7cf`

Use learning rates 2e-5, 5e-5, or 1e-4 only. Higher LRs degrade the model rather than recovering knowledge:

| LR | WMDP Step 0 | WMDP Step 500 | Recovery | MMLU Step 0 | MMLU Step 500 |
|----|-------------|---------------|----------|-------------|---------------|
| 2e-5 | 34.6% | 33.9% | -0.7 | 46.0% | 46.3% |
| 5e-5 | 34.6% | 35.3% | +0.7 | 46.0% | 46.3% |
| 1e-4 | 34.6% | 35.1% | +0.6 | 46.0% | 44.1% |
| 2e-4 | 34.6% | 30.5% | -4.0 | 46.0% | 37.9% |

(Above results with constant LR, no warmup, epochs=5, eval_every=100.)

Default LR: 5e-5 (most WMDP recovery at step 500).

Output dir: `runs/tamper_filtered_{TAG}`

Settings: `--max_steps=3000 --eval_every=500 --lr={LR} --lr_scheduler_type={SCHEDULE} {WARMUP_ARG} --eval_mmlu`

Use the same 6 sweep configs as the LoRA rank models, but replace the LR column with the filtered model default (5e-5) for non-LR-sweep configs.

## Follow-up: Seed Sweep for Promising Configs

For promising configs:
1. Re-run with `--eval_cloze_prob` to get both MCQA and Cloze metrics
2. Test 5 random seeds (1-5) using `--seed={N}`
3. Output dir: `runs/tamper_r{RANK}_{TAG}_seed{N}`
