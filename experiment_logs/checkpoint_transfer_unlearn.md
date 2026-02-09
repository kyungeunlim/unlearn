# Checkpoint Transfer Hyperparameter Tuning

**Goal:** Find the boundary between unlearning both evals and unlearning neither.

**Fixed settings:**
- `num_train_examples`: 1024
- `pdbs`: 4 (per device batch size)
- `affine`: True (loaded from EleutherAI/affine-checkpoint-transfer)
- `KL divergence retain`: True
- `lr`: 1e-3
- `layers`: [5, 10, 15, 20, 25, 30]

## Checkpoint Sweep

Finding the latest checkpoint with near-random WMDP Bio performance:

| Checkpoint | WMDP Robust | MMLU | Notes |
|------------|-------------|------|-------|
| global_step38144 | 24.19% | 27.93% | Current source |
| global_step50064 | 25.23% | 28.77% | ~Random |
| global_step54832 | 26.50% | 29.08% | ~Random (latest viable) |
| global_step57216 | - | - | Learning starts |

Affine transforms for global_step54832 available at `EleutherAI/affine-checkpoint-transfer-step54832`.

## Notes

- Higher remove_coef = stronger push towards checkpoint activations (more unlearning)
- Higher retain_coef = stronger KL divergence preservation (less capability loss)
- Looking for the sweet spot where WMDP drops but MMLU stays high, or alternatively failure spot where both drop but not all the way to 25% acc/random.

## Runs

| Job ID | retain_coef | remove_coef | Steps | retain_kl_loss | cb_loss | WMDP Bio | WMDP Robust | MMLU STEM (deprecated) | MMLU |
|--------|-------------|-------------|-------|----------------|---------|----------|-------------|-----------|------|
| Baseline | - | - | - | - | - | 42.97% | 42.97% | 36.85% | 45.10% |
| 2021158 | 2 | 5 | 256 | 0.0190 | 0.8912 | 0.3952 | - | 0.3619 | - |
| 2021163 | 2 | 20 | 256 | 0.0481 | 0.8683 | 0.3721 | - | 0.3524 | - |
| 2021160 | 5 | 10 | 256 | 0.0153 | 0.8692 | 0.3744 | - | 0.3552 | - |
| 2024175 | 5 | 2 | 256 | 0.0053 | 0.8858 | 0.3952 | - | 0.3647 | - |
| 2024177 | 5 | 1 | 256 | 0.0031 | 0.8890 | 0.4067 | - | 0.3619 | - |
| 2024176 | 10 | 5 | 256 | 0.0063 | 0.8807 | 0.3952 | - | 0.3635 | - |
| 2024178 | 20 | 10 | 256 | 0.0056 | 0.8725 | 0.3906 | - | 0.3612 | - |
| 2021162 | 10 | 20 | 256 | 0.0171 | 0.8710 | 0.3744 | - | 0.3612 | - |
| 2025807 | 2 | 30 | 256 | 0.0593 | 0.8750 | 0.3687 | - | 0.3457 | - |
| 2025808 | 2 | 40 | 256 | 0.0641 | 0.8875 | 0.3629 | - | 0.3432 | - |
| 2025809 | 2 | 50 | 256 | 0.0737 | 0.8725 | 0.3687 | - | 0.3454 | - |
| 2025810 | 2 | 60 | 256 | 0.0697 | 0.8791 | 0.3606 | - | 0.3416 | - |
| 2026007 | 2 | 100 | 256 | 0.0660 | 0.8843 | 0.3606 | - | 0.3416 | - |
| 2026828 | 2 | 100 | 512 | 0.0608 | 0.7283 | 0.3260 | - | 0.3365 | - |
| 2028287 | 2 | 100 | 2048 | 0.0498 | 0.7037 | 0.2972 | 0.3318 | 0.3222 | 0.3789 |
| 2028225 | 2 | 50 | 2048 | 0.0375 | 0.7043 | 0.2995 | - | 0.3219 | 0.3818 |
| 2038116 | 5 | 5 | 512 | 0.0062 | 0.7423 | 0.3744 | 0.3767 | 0.3590 | 0.4296 |
| 2038117 | 5 | 20 | 512 | 0.0151 | 0.7385 | 0.3479 | - | 0.3479 | 0.4208 |
| 2038118 | 20 | 5 | 512 | 0.0022 | 0.7441 | 0.3813 | **0.3848** | 0.3641 | **0.4375** |

### Batch 1 Observations
- All configurations produce similar WMDP (0.37-0.40) and MMLU (0.35-0.36)
- Higher remove_coef with low retain_coef (2,20) slightly worse MMLU
- Need baseline model performance for comparison
- Need to explore more extreme configs to find boundary

### Batch 2 Observations
- Even with lowest remove_coef (1), WMDP only rises to 0.4067 (vs 0.39-0.40 for others)
- All configs converge to similar WMDP (0.39-0.41) and MMLU (0.36-0.37)
- Unlearning effect seems to plateau early; diminishing returns from higher remove_coef
- Need baseline model performance to understand "no unlearning" case

### Batch 3 Observations (retain=2, varying remove)
- MMLU drops steadily as remove_coef increases: 0.36 → 0.34
- WMDP also drops: 0.40 → 0.36
- remove=60 and remove=100 give identical results (plateau)
- retain_kl_loss increases with higher remove_coef (0.06-0.07 vs 0.02-0.05 earlier)

### 2x Steps (retain=2, remove=100, 512 steps)
- WMDP drops further: 0.3606 → 0.3260 (closer to checkpoint baseline of 0.2419)
- MMLU drops slightly: 0.3416 → 0.3365 (still above checkpoint baseline of 0.2778)
- cb_loss drops significantly: 0.8843 → 0.7283 (model moving closer to checkpoint)

### 8x Steps (retain=2, 2048 steps)
- Both remove=50 and remove=100 converge to nearly identical results
- WMDP: ~0.30 (close to checkpoint baseline of 0.2419)
- MMLU: ~0.32 (still above checkpoint baseline of 0.2778)
- cb_loss: ~0.70 (model very close to checkpoint activations)
- **Conclusion**: More training steps is the key to unlearning, not higher remove_coef

## Retain Coef Sweep (512 steps, remove=5)

**Goal:** Find minimum retain_coef that maintains MMLU within 1% of baseline (0.4375).

**Fixed settings:**
- remove_coef: 5
- Steps: 512 (num_train_examples=2048, pdbs=4)
- affine: loaded from EleutherAI/affine-checkpoint-transfer

**MMLU threshold:** >= 0.4275 (1% below baseline of 0.4375)

| Job ID | retain_coef | remove_coef | Steps | retain_kl_loss | cb_loss | WMDP Robust | MMLU |
|--------|-------------|-------------|-------|----------------|---------|-------------|------|
| Baseline | - | - | - | - | - | 42.97% | 45.10% |
| 2042137 | 15 | 5 | 512 | 0.0027 | 0.7393 | 0.3894 | 0.4322 |
| 2042650 | 10 | 5 | 512 | 0.0037 | 0.7392 | **0.3779** | **0.4312** |
| 2042651 | 5 | 5 | 512 | 0.0062 | 0.7397 | 0.3779 | 0.4302 |
| 2042935 | 2 | 5 | 512 | 0.0125 | 0.7477 | 0.3698 | 0.4235 |
| 2042936 | 1 | 5 | 512 | 0.0200 | 0.7386 | 0.3721 | 0.4202 |
| 2043897 | 1 | 12 | 512 | - | - | 0.3468 | 0.4143 |

## Tampering Attack Results

| Model | Starting WMDP | After 10 steps | Final (110 steps) | Notes |
|-------|---------------|----------------|-------------------|-------|
| ret5_rm5_512 | 37.7% | ~42% | ~43% | Full recovery by step 30 |
| ret1_rm5_512 | 37.2% | 41.9% | 42.9% | Full recovery by step 10 |
| LoRA ret5_rm5_r16 | 32.5% | 42.3% | 42.5% (step 70) | Full recovery by step 5 |

**Conclusion:** Unlearning is easily reversed with small amount of finetuning on bio data. Both merged and LoRA models recover to baseline (~43%) within 5-30 finetuning steps. Cloze probability also recovers (0.19 → 0.24).

## KL Retain Loss Epoch Sweep

**Goal:** Find the boundary between minimal capability impact and noticeable damage, varying training length.

**Settings** (differences from fixed settings above):
- `num_train_examples`: 1024
- Batch size: 4 per device, grad accumulation 2 (effective batch 32)

### 1 Epoch (32 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Robust | MMLU |
|-------------|-------------|----------------|---------|-------------|------|
| -           | -           | -              | -       | 42.97%      | 45.10% |
| 1           | 3           | 0.247          | 6.30    | 29.95%      | 33.64% |
| 5           | 5           | 0.156          | 6.31    | 31.80%      | 35.81% |

### 10 Epochs (320 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Bio (↓) | MMLU |
|-------------|-------------|----------------|---------|--------------|------|
| -           | -           | -              | -       | 42.97%       | 45.10% |
| 5           | 8           | 0.025          | 3.21    | 30.99%       | 34.73% |

### 100 Epochs (3200 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Robust | MMLU |
|-------------|-------------|----------------|---------|-------------|------|
| -           | -           | -              | -       | 42.97%      | 45.10% |
| 5           | 8           | 0.022          | 2.44    | 29.38%      | 31.66% |


### Epoch Sweep Observations
- **Longer training significantly reduces losses**: cb_loss drops from ~6.3 (1ep) to ~4.1 (4ep) to ~3.2 (10ep)
- **retain_kl_loss decreases with more training**: 0.15-0.25 → 0.04-0.09 → 0.025
- **However, longer training does not improve unlearning** - WMDP Bio stays at ~30-31% regardless
- **10 epochs (320 steps)**: WMDP Bio 30.99%, MMLU 34.87% - no improvement over 1 epoch

### Epoch Sweep Conclusions
1. **KL divergence retain loss works** but unlearning incomplete at all settings
2. **1 epoch is sufficient** - longer training does not improve unlearning

All models saved to `models/EleutherAI/deep-ignorance-unfiltered_kl_ret{X}_rm{Y}[_ep4]`.

## LoRA (adapter saved separately, not merged)

**Base config** from retain coef sweep sweet spot (retain=5, remove=5, 512 steps).

**LoRA settings:**
- `lora_r`: 16
- `lora_alpha`: 16
- `lora_dropout`: 0.05

| Job ID | retain_coef | remove_coef | lora_r | Steps | retain_kl_loss | cb_loss | WMDP Robust | MMLU |
|--------|-------------|-------------|--------|-------|----------------|---------|-------------|------|
| Baseline | - | - | - | - | - | - | 42.97% | 45.10% |
| 2133536 | 5 | 5 | 16 | 512 | 0.0074 | 0.8748 | 0.3272 | 0.3856 |
| 2133607 | 2 | 2000 | 16 | 512 | 0.2584 | 0.8830 | 0.3203 | 0.2971 |
| 2133686 | 2 | 5000 | 16 | 512 | 0.2646 | 0.8880 | 0.3122 | 0.2980 |
| 2133700 | 0 | 2000 | 16 | 512 | 0.0000 | 0.8717 | 0.3134 | 0.2901 |
| 2133707 | 0 | 2000 | 8 | 512 | 0.0000 | 0.8807 | 0.3018 | 0.2876 |
| 2133708 | 0 | 2000 | 32 | 512 | 0.0000 | 0.8767 | 0.3145 | 0.2975 |
| 2133741 | 0 | 2000 | 64 | 512 | 0.0000 | 0.8881 | 0.3191 | 0.3029 |
| 2133712 | 0 | 2000 | full | 512 | 0.0000 | 31.1447 | 0.2650 | 0.2300 |
| 2133747 | 0 | 2000 | 16 (muon) | 512 | cancelled | | | |
| 2133758 | 0 | 2000 | full (muon*) | 512 | 0.0000 | diverged | 0.2350 | 0.2464 |

\* Muon had a bug: `size(0) < 10000` threshold caused QKV (12288) and MLP up (16384) weights to route to AdamW. Only 64/388 trainable 2D params used Muon. cb_loss diverged (2.09 → 761K). Fixed threshold to `< 50000`.

| Job ID | retain_coef | remove_coef | lora_r | Steps | retain_kl_loss | cb_loss | WMDP Robust | MMLU |
|--------|-------------|-------------|--------|-------|----------------|---------|-------------|------|
| 2133765 | 0 | 2000 | full (muon) | 512 | 0.0000 | diverged | - | - |
| 2133784 | 0 | 2000 | full (muon) | 512 | 0.0000 | diverged | - | - |
| 2133785 | 0 | 2000 | full (muon, lr=5e-4) | 512 | 0.0000 | 0.5941 | 0.2903 | 0.2849 |

## Unconstrained Unlearning: Long Tamper Resistance

Goal: Test whether unconstrained unlearning (retain_coef=0) produces tamper-resistant models. 30 epoch finetune attack (512 bio examples, lr=2e-5, eval every 100 steps).

### ct_fullrank_ret0_rm2000 (SFT AdamW unlearn)

Starting WMDP: 26.5%, Starting MMLU: 23.0%

| Tamper Optimizer | Step 0 WMDP | Step 100 WMDP | Step 500 WMDP | Step 1000 WMDP | Final WMDP | Final MMLU | Status |
|------------------|-------------|---------------|---------------|----------------|------------|------------|--------|
| AdamW (short)    | 26.5%       | 26.6%         | -             | -              | 26.6% (step 110) | - | Complete |
| AdamW (30ep)     | -           | -             | -             | -              | -          | -          | Running |
| Muon (30ep)      | -           | -             | -             | -              | -          | -          | Running |

### ct_sft_muon_ret0_rm2000 (Muon unlearn)

Starting WMDP: 29.0%, Starting MMLU: 29.3%

| Tamper Optimizer | Step 0 WMDP | Step 100 WMDP | Step 500 WMDP | Step 1000 WMDP | Final WMDP | Final MMLU | Status |
|------------------|-------------|---------------|---------------|----------------|------------|------------|--------|
| Muon (short)     | 29.0%       | 28.1%         | 28.6%         | 27.9%          | 28.8% (step 1100) | - | Complete |
| AdamW (old long) | 29.0%       | 30.2%         | 30.0%         | 29.5%          | 28.9% (step 5500) | 35.9% | Complete |
| AdamW (30ep)     | 29.0%       | 29.7%         | -             | -              | -          | -          | Running |
| Muon (30ep)      | -           | -             | -             | -              | -          | -          | Running |

### random_init (control baseline)

Starting WMDP: 26.2%, Starting MMLU: 24.5%

| Tamper Optimizer | Step 0 WMDP | Step 100 WMDP | Step 500 WMDP | Step 1000 WMDP | Final WMDP | Final MMLU | Status |
|------------------|-------------|---------------|---------------|----------------|------------|------------|--------|
| AdamW (30ep)     | 26.2%       | 27.1%         | -             | -              | -          | -          | Running |
