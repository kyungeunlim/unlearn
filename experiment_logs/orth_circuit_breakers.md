# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Metric Interpretation

**orth_loss**: Average pairwise cosine similarity between forget item representations in each batch. Measures whether forget items collapse to the same direction or remain diverse.
- **~1.0** = bad (all forget items aligned to same direction, vulnerable to single-direction attacks)
- **~0.5** = moderate diversity
- **~0** = good (orthogonal forget directions)

## Results

| remove | orth | retain | steps | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|-------------|------|-------------|---------|-----------|-------|
| -      | -    | -      | -     | 42.97%    | 45.10% | -           | -       | -         | Baseline |
| 23     | 5    | 0      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 10   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 20   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 0    | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned, orth=0 control |
| 23     | 10   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned, keyword mask (type: regex) |

### Orth Loss Fix Validation (LoRA r=16, orth ramp + seq_len scaling)

Goal: Validate orth loss improvements. Changes: orth_coeff ramps 0 to full over training (was decaying), orth_loss scaled by mean seq_len to compensate for mean-pooling gradient dilution. All runs: remove=23, retain=2, r=16, 32 steps, 1024 examples.

| remove | orth | retain | steps | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|-------------|------|-------------|---------|-----------|-------|
| -      | -    | -      | -     | 42.97%      | 45.10% | -         | -       | -         | Baseline |
| 23     | 5    | 2      | 32    | 37.44%      | 43.75% | 21.77     | 0.51    | 730.0     | orth_loss now seq_len-scaled |
| 23     | 20   | 2      | 32    | 38.48%      | 43.65% | 29.64     | 0.49    | 932.5     | |
| 23     | 50   | 2      | 32    | 37.90%      | 43.55% | 31.11     | 0.63    | 986.4     | |

## SFT

| remove | orth | retain | steps | WMDP | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|------|-------------|------|-------------|---------|-----------|-------|
| 23     | 10   | 2      | 32    | -      | -         | -           | -       | -         | SFT planned, keyword mask (type: regex) |
| 30     | 15   | 2      | 32    | 26.55% | -         | -    | 4.30        | 0.18    | 0.99      | |
| 30     | 15   | 2      | 512   | 24.12% | 25.81%    | 23.76% | 1.08        | 0.015   | 0.05      | Severe capability damage |
| 40     | 20   | 2      | 32    | 26.63% | -         | -    | -           | -       | -         | Log truncated |
| 15     | 15   | 2      | 512   | 23.88% | 25.35%    | 29.90% | 1.01        | 0.03    | 0.04      | Capability damage |
| 30     | 100  | 2      | 32    | 39.2%  | -         | -    | 3.52        | 0.35    | 0.33      | Preserved capability, worse unlearning |
| 30     | 100  | 2      | 256   | 29.85% | 28.34%    | 42.18% | 2.95        | 0.10    | 0.09      | Good balance |
| 30     | 15   | 2      | 64    | 27.73% | -         | -    | 3.46        | 0.06    | 0.52      | Good balance, near-baseline MMLU |
| 30     | 15   | 2      | 128   | 24.35% | -         | -    | 2.15        | 0.02    | 0.25      | Better orth, some capability loss |
| 15     | 15   | 15     | 512   | 28.28% | 28.11%    | 43.53% | 0.40        | 0.07    | 0.13      | Good unlearning + MMLU preserved |
| 15     | 15   | 50     | 512   | 35.90% | **33.64%**    | **44.81%** | 0.26        | 0.09    | 0.25      | Better MMLU, weaker unlearning |


## Tamper Resistance: Orthfix (retain=2, r=16)

Finetune attack (512 examples, lr=2e-5, eval every 5 steps) on the orth loss fix validation models above.

| Model | Step 0 | Step 5 | Step 10 | Step 25 | Step 50 | Notes |
|-------|--------|--------|---------|---------|---------|-------|
| Baseline | 42.97% | - | - | - | - | Original model |
| orth=5, ret=2 | 37.67% | 42.40% | 39.75% | 41.24% | 43.66% | Full recovery by step 5 |
| orth=20, ret=2 | 38.71% | 41.71% | 41.36% | 42.74% | 44.24% | Full recovery by step 5 |
| orth=50, ret=2 | 38.13% | 41.24% | 41.01% | 42.74% | 44.47% | Full recovery by step 5 |

Plots: `experiment_logs/tamper_orthfix_ret2_mcqa.png`, `experiment_logs/tamper_orthfix_ret2_cloze.png`

# Unrestrained Unlearning with Low Rank Updates

## Unrestrained Unlearning LoRA Rank Sweep (retain_coef=0)

Goal: Determine minimum LoRA rank needed for tamper-resistant unlearning with orth circuit breakers. All runs: remove=23, orth=5, retain=0, 32 steps, 1024 examples.

| rank | WMDP Robust | MMLU | cb_loss | orth_loss | Tamper WMDP (step 105) | Tamper Cloze (step 105) | Notes |
|------|-------------|------|---------|-----------|------------------------|-------------------------|-------|
| -    | 42.97%      | 45.10% | -     | -         | -                      | -                       | Baseline |
| 8    | 26.73%      | 22.95% | 0.0007 | 0.9999   | 26.73%                 | 0.141                   | |
| 16   | 26.73%      | 22.95% | 0.0007 | 0.9999   | 26.73%                 | 0.143                   | |
| 32   | 26.73%      | 22.95% | 0.0008 | 0.9999   | 26.73%                 | 0.143                   | |
| 64   | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.142                   | |
| 128  | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.144                   | |
| 256  | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.144                   | |
| 512  | 26.73%      | 22.95% | 0.0010 | 0.9999   | 26.73%                 | 0.144                   | |

## Long Tamper Resistance: LoRA Rank Sweep (retain=0)

Finetune attack (512 examples, lr=2e-5, 30 epochs, eval every 100/500 steps) on retain=0 LoRA rank sweep models. All unlearned with AdamW (unless noted), remove=23, orth=5, retain=0, 1024 examples, all 32 layers. Jobs cancelled at step 500 (attn/mlp) or step 1600-1700 (all-module).

### attn-only LoRA (AdamW tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Notes |
|------|--------|----------|-------|
| - | 42.97 / 45.10 | - | Baseline |
| 8 | 23.5 / 24.7 | - | Last step 400: 44.0 / 42.5 |
| 16 | 26.7 / 23.0 | 32.1 / 41.2 | |
| 32 | 26.7 / 23.0 | 26.6 / 24.8 | |
| 64 | 26.7 / 23.0 | 25.2 / 24.6 | |
| 128 | 23.5 / 24.7 | 24.5 / 24.8 | |
| 256 | 24.1 / 26.9 | 26.6 / 23.0 | |
| 512 | 24.1 / 26.9 | 27.1 / 23.0 | |

### mlp-only LoRA (AdamW tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Notes |
|------|--------|----------|-------|
| - | 42.97 / 45.10 | - | Baseline |
| 8 | 26.7 / 23.0 | 42.2 / 45.7 | |
| 16 | 26.7 / 23.0 | 34.8 / 41.5 | |
| 32 | 26.7 / 23.0 | 27.0 / 26.0 | |
| 64 | 23.5 / 24.7 | 24.9 / 23.9 | |
| 128 | 25.7 / 25.5 | 27.5 / 26.2 | |
| 256 | 23.9 / 25.8 | 26.7 / 23.0 | |
| 512 | 23.5 / 24.7 | 26.7 / 23.0 | |

### all-module LoRA (AdamW tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Step 1000 | Step 1500 | Notes |
|------|--------|----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | Baseline |
| 8 | 26.2 / 25.5 | 31.8 / 40.3 | - | - | |
| 16 | 26.7 / 23.0 | 28.0 / 25.0 | - | - | |
| 32 | 26.7 / 23.0 | 26.0 / 25.9 | 25.7 / 25.6 | 26.4 / 24.8 | Step 1600: 26.4 / 24.8 |
| 64 | 26.7 / 23.0 | 26.5 / 23.7 | 26.7 / 26.3 | 27.3 / 26.4 | Step 1600: 28.1 / 26.2 |
| 128 | 25.8 / 26.8 | 26.7 / 23.0 | 26.4 / 24.4 | 24.0 / 24.2 | Step 1700: 22.4 / 24.6 |
| 256 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | |
| 512 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | |

### all-module LoRA (Muon tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Step 1000 | Notes |
|------|--------|----------|-----------|-------|
| 8 | 26.2 / 25.5 | 26.7 / 23.0 | - | |
| 12 (Muon unlearn) | 28.9 / 29.9 | 35.0 / 40.7 | 34.5 / 42.4 | |
| 16 | 26.7 / 23.0 | 24.7 / 24.7 | - | |

### r12 LoRA unlearn comparison (Muon tamper) — WMDP / MMLU

| target | unlearn opt | Step 0 | Step 500 | Step 1000 | Notes |
|--------|-------------|--------|----------|-----------|-------|
| attn | Muon | 32.7 / 38.9 | 38.0 / 43.7 | 39.6 / 44.5 | |
| mlp | Muon | 34.2 / 41.1 | 40.7 / 45.3 | 40.7 / 45.1 | |
| all | Muon | 28.9 / 29.9 | 35.0 / 40.7 | 34.5 / 42.4 | |
| all | AdamW | 28.5 / 34.9 | 38.7 / 44.6 | - | |

### SFT (full-rank) unlearn tamper

| Model | Tamper Opt | Step 0 | Step 500 | Step 1000 | Notes |
|-------|-----------|--------|----------|-----------|-------|
| orth Muon SFT (rm23, orth10, ret0) | AdamW | 27.1 / - | 25.5 / - | 25.8 / - | Complete (step 1100: 26.0) |
| orth AdamW SFT (rm23, orth5, ret0) | AdamW | - | - | - | Failed (model path error) |
| orth AdamW SFT (rm23, orth5, ret0) | Muon | - | - | - | Failed (model path error) |

### Layer exclusion ablation (AdamW tamper, attn r=16) — WMDP / MMLU

| Excluded layer | Step 0 | Step 200 | Step 400 | Notes |
|----------------|--------|----------|----------|-------|
| L0 | 37.8 / 45.8 | 43.6 / 46.7 | 44.0 / 47.5 | |
| L16 | 39.2 / 46.2 | 43.4 / 46.9 | 43.6 / 47.7 | |
| L31 | 37.4 / 45.7 | 43.2 / 46.7 | 43.8 / 47.7 | |

### Old attn-only tamper (50 epochs, eval every 100 steps, no MMLU)

| rank | Step 0 | Step 200 | Step 500 | Step 800 | Step 1000 |
|------|--------|----------|----------|----------|-----------|
| 8 | 26.73 | 26.50 | 39.75 | 42.40 | 41.24 |
