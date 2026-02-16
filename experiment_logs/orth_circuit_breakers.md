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

Goal: Determine minimum LoRA rank needed for tamper-resistant unlearning with orth circuit breakers. All runs: remove=23, orth=5, retain=0, 32 steps, 1024 examples. Trained with pdbs=1 so orth loss was always 0 (pairwise loss requires batch_size >= 2). Effectively cb_loss only.

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

Finetune attack (512 examples, lr=2e-5, 30 epochs, eval every 100/500 steps) on retain=0 LoRA rank sweep models. All unlearned with AdamW (unless noted), remove=23, orth=5, retain=0, 1024 examples, all 32 layers. Jobs cancelled at step 500 (attn/mlp) or step 1600-1700 (all-module). Unlearned models used pdbs=1 so orth loss was always 0 (cb_loss only), except r12 which used pdbs=4 (orth loss active).

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

| rank | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3300 | Notes |
|------|--------|----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | Baseline |
| 8 | 26.2 / 25.5 | 29.8 / 38.1 | 31.8 / 40.6 | 29.1 / 39.3 | 29.3 / 39.2 | Peaks step 800, settles ~29 |
| 16 | 26.7 / 23.0 | 27.1 / 25.3 | 29.0 / 24.9 | 29.1 / 24.8 | 29.1 / 24.6 | Gradual climb to ~29 |
| 32 | 26.7 / 23.0 | 26.0 / 25.9 | 25.7 / 25.6 | 25.5 / 25.1 | 25.8 / 25.2 | Flat through 3300 |
| 64 | 26.7 / 23.0 | 26.5 / 23.7 | 26.7 / 26.3 | 26.8 / 26.3 | 27.8 / 26.5 | Flat through 3300 |
| 128 | 25.8 / 26.8 | 26.7 / 23.0 | 26.4 / 24.4 | 24.1 / 24.4 | 23.6 / 24.7 | Flat/declining through 3300 |
| 256 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | - | Cancelled step 500, resubmitted |
| 512 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | - | Cancelled step 500, resubmitted |

### all-module LoRA (Muon tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3300 | Notes |
|------|--------|----------|-----------|-----------|-----------|-------|
| 8 | 26.2 / 25.5 | 26.7 / 23.0 | 26.4 / 24.5 | 25.1 / 25.3 | 23.8 / 25.3 | Flat through 3300 |
| 12 (Muon unlearn) | 28.9 / 29.9 | 35.0 / 40.7 | 34.4 / 42.4 | 34.8 / 41.2 | 31.9 / 39.7 | pdbs=4 (orth loss active), cracked then declining |
| 12 (AdamW unlearn) | 28.5 / 34.9 | 38.7 / 44.6 | 37.8 / 45.8 | 35.2 / 44.9 | 34.9 / 43.8 | pdbs=4 (orth loss active), cracked, slow decline |
| 16 | 26.7 / 23.0 | 24.6 / 24.7 | 23.5 / 24.6 | 26.2 / 23.4 | 26.6 / 23.2 | Flat through 3300 |

### r12 LoRA unlearn comparison (Muon tamper) — WMDP / MMLU

r12 models unlearned with pdbs=4 (orth loss active). Not directly comparable to r8/r16 rank sweep models above which used pdbs=1 (orth loss always 0).

| target | unlearn opt | Step 0 | Step 500 | Step 1000 | Step 3300 | Notes |
|--------|-------------|--------|----------|-----------|-----------|-------|
| attn | Muon | 32.7 / 38.9 | 38.0 / 43.7 | 39.6 / 44.5 | - | Completed step 1000 |
| mlp | Muon | 34.2 / 41.1 | 40.7 / 45.3 | 40.7 / 45.1 | - | Completed step 1000 |
| all | Muon | 28.9 / 29.9 | 35.0 / 40.7 | 34.4 / 42.4 | 31.9 / 39.7 | |
| all | AdamW | 28.5 / 34.9 | 38.7 / 44.6 | 37.8 / 45.8 | 34.9 / 43.8 | |

### SFT (full-rank) unlearn tamper

| Model | Tamper Opt | Step 0 | Step 500 | Step 1000 | Notes |
|-------|-----------|--------|----------|-----------|-------|
| orth Muon SFT (rm23, orth10, ret0) | AdamW | 27.1 / - | 25.5 / - | 25.8 / - | Complete (step 1100: 26.0) |
| orth AdamW SFT (rm23, orth5, ret0) | AdamW | - | - | - | Resubmitted (path fix) |
| orth AdamW SFT (rm23, orth5, ret0) | Muon | - | - | - | Resubmitted (path fix) |

### Layer exclusion ablation (AdamW tamper, attn-only r=16) — WMDP / MMLU

| Excluded layer | Step 0 | Step 200 | Step 400 | Notes |
|----------------|--------|----------|----------|-------|
| L0 | 37.8 / 45.8 | 43.6 / 46.7 | 44.0 / 47.5 | attn-only LoRA, not useful |
| L16 | 39.2 / 46.2 | 43.4 / 46.9 | 43.6 / 47.7 | attn-only LoRA, not useful |
| L31 | 37.4 / 45.7 | 43.2 / 46.7 | 43.8 / 47.7 | attn-only LoRA, not useful |

### Layer exclusion ablation (AdamW tamper, all-module r=16) — WMDP / MMLU

| Excluded layer | Step 0 | Step 200 | Step 400 | Step 600 | Step 800 | Step 1000 | Step 1200 | Step 1400 | Step 1600 | Step 1800 | Step 2000 | Notes |
|----------------|--------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | - | - | - | - | - | - | Baseline |
| L0 | 27.65 / 32.37 | 41.36 / 45.72 | 43.09 / 46.80 | 43.66 / 46.89 | 44.01 / 47.14 | 43.43 / 46.82 | 42.63 / 46.61 | 41.59 / 46.33 | 40.55 / 46.19 | 41.36 / 46.20 | 41.24 / 46.06 | Running |
| L16 | 26.96 / 32.07 | 41.13 / 45.73 | 42.97 / 46.83 | 43.78 / 46.62 | 42.86 / 46.94 | 41.59 / 46.70 | 42.17 / 46.45 | 41.13 / 46.18 | 39.40 / 45.83 | 39.29 / 45.78 | 39.75 / 45.58 | Running |
| L31 | 25.92 / 34.61 | 40.90 / 45.69 | 42.63 / 46.69 | 43.09 / 46.87 | 43.32 / 46.98 | 42.86 / 46.60 | 41.13 / 46.45 | 41.01 / 46.47 | 40.09 / 46.08 | 39.75 / 46.00 | 39.52 / 45.98 | Running |

### Weight interpolation attack — WMDP / MMLU

model_interp = (1 - alpha) * unlearned + alpha * base. No training, just weight blending. Unlearned models: r=8 and r=16, all-module, pdbs=1 (orth loss was always 0 during unlearning).

| Model | Alpha 0.1 | Alpha 0.3 | Alpha 0.5 | Alpha 0.7 | Alpha 0.9 | Alpha 0.95 |
|-------|-----------|-----------|-----------|-----------|-----------|------------|
| r=8 | 25.7 / 25.5 | 25.7 / 25.5 | 24.2 / 26.1 | 25.7 / 25.5 | 42.7 / 46.9 | 43.1 / 47.6 |
| r=16 | 26.7 / 23.0 | 26.7 / 23.0 | 26.7 / 23.0 | 26.7 / 23.0 | 37.4 / 44.4 | 42.1 / 47.3 |

### 500-step tamper sweep on r=8 all-module (pdbs=1) — WMDP / MMLU

Single eval at step 500. Unlearned model: r=8, all-module, pdbs=1.

| Attack config | Step 500 WMDP | Step 500 MMLU | Notes |
|---------------|---------------|---------------|-------|
| full_lr2e-5 | 27.0% | 26.5% | |
| full_lr2e-4 | 32.8% | 36.8% | Partial crack |
| full_lr5e-4 | 27.9% | 24.0% | Too high LR, diverged |
| full_lr1e-3 | 27.4% | 23.7% | Too high LR, diverged |
| lora_r8_lr1e-3 | 26.7% | 23.0% | |
| lora_r16_lr1e-3 | 26.7% | 23.0% | |

### Repair-then-attack on r=8 all-module (pdbs=1) — WMDP / MMLU

Phase 1: WikiText repair (200 steps, lr=2e-5). Phase 2: Bio attack (500 steps, lr=2e-5).

| Phase | Step | WMDP | MMLU |
|-------|------|------|------|
| repair | 0 | 26.2% | 25.5% |
| repair | 100 | 24.5% | 23.4% |
| repair | 200 | 24.8% | 26.3% |
| attack | 0 | 24.8% | 26.3% |

### Empirical rank of weight deltas (99% energy threshold)

| Model | Total Frob | QKV Mean/Max Rank | Dense Mean/Max | MLP Up Mean/Max | MLP Down Mean/Max |
|-------|-----------|-------------------|----------------|-----------------|-------------------|
| r=8 (AdamW, pdbs=1) | 16.9 | 4.1 / 6 | 2.0 / 4 | 2.5 / 4 | 1.0 / 2 |
| r=12 (Muon, pdbs=4) | 11.5 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 |
| r=12 (AdamW, pdbs=4) | 14.4 | 11.0 / 12 | 10.6 / 12 | 10.5 / 12 | 7.1 / 12 |
| r=16 (AdamW, pdbs=1) | 26.2 | 5.1 / 8 | 2.6 / 7 | 3.3 / 7 | 1.0 / 1 |
| r=32 (AdamW, pdbs=1) | 38.7 | 6.6 / 13 | 3.4 / 10 | 5.1 / 10 | 1.0 / 1 |

### Old attn-only tamper (50 epochs, eval every 100 steps, no MMLU)

Tamper on rank sweep models (pdbs=1, orth loss was always 0 during unlearning).

| rank | Step 0 | Step 200 | Step 500 | Step 800 | Step 1000 |
|------|--------|----------|----------|----------|-----------|
| 8 | 26.73 | 26.50 | 39.75 | 42.40 | 41.24 |

### all-module LoRA lr=2e-4 tamper (AdamW, 32 epochs, eval every 500 steps) — WMDP / MMLU

Unlearned models: pdbs=1 (orth loss always 0), remove=23, orth=5, retain=0, all 32 layers.

| rank | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3520 | Notes |
|------|--------|----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | Baseline |
| 1 | - | - | - | - | - | Running |
| 2 | 24.5 / 24.7 | 30.7 / 35.6 | - | - | - | Running |
| 4 | 24.1 / 26.9 | 25.9 / 27.8 | - | - | - | Running |
| 8 | 26.2 / 25.5 | 29.2 / 31.6 | 27.9 / 30.7 | 28.1 / 30.8 | 28.0 / 30.8 | |
| 16 | 26.7 / 23.0 | 28.0 / 29.9 | 26.7 / 30.0 | 26.3 / 30.2 | 26.7 / 30.2 | |
| 32 | 26.7 / 23.0 | 26.7 / 26.7 | 27.4 / 26.4 | 27.4 / 27.1 | 27.2 / 27.1 | |
| 64 | 26.7 / 23.0 | 26.4 / 28.9 | 27.1 / 28.8 | 26.8 / 28.9 | 27.0 / 29.1 | |
| 128 | 25.8 / 26.8 | 27.5 / 25.5 | 27.1 / 25.6 | 25.9 / 25.5 | 26.2 / 25.4 | |
| 256 | 24.1 / 26.9 | 24.2 / 26.2 | 23.0 / 25.2 | 23.5 / 25.3 | 24.1 / 25.3 | |
| 512 | 24.1 / 26.9 | 24.1 / 24.6 | 28.6 / 25.1 | 27.8 / 25.1 | 27.5 / 25.2 | |

### deep-ignorance-e2e-strong-filter tamper (AdamW, 512 examples, 5 epochs, eval every 100 steps) — WMDP / MMLU

| lr | Step 0 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | Step 550 |
|----|--------|----------|----------|----------|----------|----------|----------|
| 2e-5 | 34.56 / 46.00 | 33.29 / 45.74 | 33.64 / 46.18 | 34.22 / 46.23 | 33.87 / 46.47 | 33.87 / 46.34 | 33.29 / 46.31 |
| 5e-5 | 34.56 / 46.00 | 32.37 / 46.14 | 33.64 / 45.75 | 34.79 / 46.35 | 34.79 / 46.27 | 35.25 / 46.33 | 34.33 / 46.45 |
| 1e-4 | 34.56 / 46.00 | 34.33 / 44.50 | 34.45 / 44.27 | 34.33 / 44.00 | 35.02 / 44.17 | 35.14 / 44.11 | 35.02 / 44.25 |
| 2e-4 | 34.56 / 46.00 | 30.76 / 37.23 | 31.68 / 35.97 | 30.18 / 37.82 | 29.61 / 38.04 | 30.53 / 37.86 | 31.11 / 37.95 |

### r=8 all-module tamper variants (AdamW, 3500 steps, eval every 500) — WMDP / MMLU

Unlearned model: r=8, all-module, pdbs=1 (orth loss always 0). Baseline: 42.97 / 45.10.

| Attack config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 |
|---------------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| cosine, lr=2e-5 | 26.96 / 26.18 | 31.91 / 38.20 | 29.49 / 37.52 | 28.00 / 37.12 | 28.11 / 36.83 | 28.23 / 36.82 | 28.34 / 36.80 | 28.34 / 36.80 |
| warmup 10%, lr=2e-5 | 26.96 / 26.18 | 32.60 / 38.01 | 28.23 / 37.07 | 29.03 / 37.12 | 28.00 / 36.60 | 28.11 / 36.68 | 28.46 / 36.48 | 28.69 / 36.31 |
| cosine + warmup 10%, lr=2e-5 | 26.96 / 26.18 | 27.42 / 25.79 | 27.88 / 30.67 | 26.50 / 29.84 | 27.76 / 29.55 | 27.53 / 29.43 | 27.19 / 29.43 | 27.19 / 29.43 |
| cosine, lr=2e-4 | 26.96 / 26.18 | 28.34 / 26.66 | 27.42 / 28.32 | 27.53 / 28.34 | 27.76 / 28.40 | 27.76 / 28.46 | 27.76 / 28.48 | 27.76 / 28.48 |
| warmup 10%, lr=2e-4 | 26.96 / 26.18 | 26.61 / 26.63 | 26.73 / 26.81 | 26.61 / 27.22 | 26.61 / 27.30 | 26.61 / 26.74 | 26.61 / 26.91 | 26.50 / 26.36 |
| cosine + warmup 10%, lr=2e-4 | 26.96 / 26.18 | 33.53 / 34.29 | 35.02 / 36.74 | 34.91 / 36.93 | 35.02 / 36.95 | 35.25 / 36.94 | 35.37 / 36.95 | 35.37 / 36.94 |

### Full-finetune tamper sweep across LoRA ranks (AdamW, 3000 steps, eval every 500) — WMDP Bio Acc (final)

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Filtered model uses LRs 2e-5/5e-5/1e-4 (vs rank sweep: 2e-5/2e-4/1e-3).

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|
| cos/wr01/lr2e-5 | **40.4** | **40.9** | 27.2 | 25.7 | **27.3** | 25.4 | 23.5 | 25.7 | 26.7 | 33.6 |
| cos/wr01/lr2e-4 (std) | 33.0 | 31.9 | **35.4** | **35.5** | 26.8 | **26.5** | **26.5** | 25.1 | 27.7 | 34.3 |
| cos/wr01/lr1e-3 | 26.6 | 25.4 | 24.6 | 26.5 | 23.6 | 25.2 | 25.8 | **27.2** | 25.1 | 34.1 |
| cos/ws30/lr2e-4 | 30.9 | 34.0 | 29.7 | 27.8 | 27.1 | 26.4 | 25.8 | 26.2 | 27.4 | **35.4** |
| cos/ws100/lr2e-4 | 34.3 | 33.6 | 30.9 | 28.1 | 26.8 | 25.5 | 23.6 | 24.6 | 25.1 | 33.6 |
| const/wr01/lr2e-4 | 32.0 | 27.8 | 26.5 | 28.2 | 25.0 | 26.2 | 25.4 | 24.3 | **27.8** | 33.1 |

### Full-finetune tamper sweep across LoRA ranks (AdamW, 3000 steps, eval every 500) — MMLU (final)

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|
| cos/wr01/lr2e-5 | **45.5** | **44.6** | 29.4 | 26.6 | 25.6 | 25.1 | 24.8 | 25.9 | 22.9 | **44.7** |
| cos/wr01/lr2e-4 (std) | 35.6 | 35.7 | **36.9** | **33.7** | **30.1** | **27.9** | 26.0 | 25.8 | **25.2** | 44.4 |
| cos/wr01/lr1e-3 | 24.6 | 25.0 | 24.5 | 25.6 | 24.5 | 25.5 | 24.2 | 23.9 | 24.5 | 43.7 |
| cos/ws30/lr2e-4 | 34.4 | 33.1 | 30.7 | 31.3 | 28.7 | 27.8 | **26.8** | **26.6** | 25.2 | 44.1 |
| cos/ws100/lr2e-4 | 34.7 | 34.5 | 31.4 | 29.3 | 26.7 | 27.2 | 25.2 | 25.4 | 24.4 | 43.8 |
| const/wr01/lr2e-4 | 30.9 | 26.4 | 26.4 | 27.5 | 26.4 | 27.8 | 24.9 | 25.5 | 25.0 | 43.8 |

### JB-CO-Text Attack: LoRA Rank Sweep (pdbs=1, ret=0, all-module)

Competing objectives jailbreak prompt wrapping WMDP-Bio MCQ questions. Template adds refusal suppression instructions before the question.

| rank | Standard WMDP | JB-CO-Text WMDP |
|------|---------------|-----------------|
| 8 | 26.15 | 26.04 |
| 16 | 26.73 | 26.73 |
| 32 | 26.73 | 26.73 |
| 64 | 26.73 | 26.73 |
| 128 | 25.81 | 24.19 |
| 256 | 24.08 | 24.08 |
| 512 | 24.08 | 24.08 |
