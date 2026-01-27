# KL Divergence Retain Loss Hyperparameter Tuning

## Goal
Find the boundary between:
1. Minimal capability impact on both retain and forget
2. Noticeable damage to both capabilities

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Checkpoint**: EleutherAI/deep-ignorance-pretraining-stage-unfiltered @ global_step38144
- **Loss type**: `--retain_kl_loss` (KL divergence on outputs)
- **Training examples**: 1024
- **GPUs**: 4x GH200
- **Batch size**: 4 per device, grad accumulation 2 (effective batch 32)
- **Learning rate**: 0.001
- **LoRA rank**: 16
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Baselines

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |
| Checkpoint (source activations) | 24.19% | 27.78% |

## Results

### 1 Epoch (32 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Bio (↓) | MMLU (↑) |
|-------------|-------------|----------------|---------|--------------|----------|
| 1           | 3           | 0.247          | 6.30    | 29.84%       | 33.61%   |
| 3           | 5           | 0.203          | 6.30    | -            | -        |
| 5           | 5           | 0.156          | 6.31    | 31.68%       | 35.73%   |
| 10          | 5           | 0.101          | 6.33    | 30.76%       | 34.98%   |
| 5           | 10          | 0.229          | 6.35    | 31.11%       | 34.55%   |
| 10          | 10          | 0.169          | 6.36    | -            | -        |
| 5           | 15          | 0.243          | 6.35    | 30.53%       | 33.81%   |
| 5           | 20          | 0.259          | 6.36    | 30.65%       | 33.68%   |
| 5           | 30          | 0.272          | 6.51    | 30.07%       | 32.40%   |

### 4 Epochs (128 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Bio (↓) | MMLU (↑) |
|-------------|-------------|----------------|---------|--------------|----------|
| 1           | 3           | 0.085          | 4.09    | 30.76%       | 35.33%   |
| 5           | 5           | 0.049          | 4.06    | 30.88%       | 34.82%   |
| 10          | 5           | 0.036          | 4.07    | 31.22%       | 35.55%   |

### 10 Epochs (320 steps)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | WMDP Bio (↓) | MMLU (↑) |
|-------------|-------------|----------------|---------|--------------|----------|
| 5           | 8           | 0.025          | 3.21    | pending      | pending  |

### Observations
- **Longer training (4 epochs) significantly reduces losses**: cb_loss drops from ~6.3 to ~4.1
- **retain_kl_loss decreases with more training**: 0.15-0.25 → 0.04-0.09
- **However, 4 epochs does NOT improve unlearning** - WMDP Bio scores are similar or slightly worse

## Key Findings

### Conclusions
1. **KL divergence retain loss works** but unlearning incomplete at all settings
2. **1 epoch is sufficient** - longer training does not improve unlearning
3. **retain_coef=1, remove_coef=3 achieves best unlearning** (29.84% WMDP Bio)
4. **Aggressive remove_coef (15-30) degrades MMLU without improving WMDP Bio**
   - remove_coef=30: WMDP Bio 30.07% (worse than ret=1,rm=3), MMLU 32.40% (2-3% degraded)
5. **WMDP Bio plateau at ~30%** - cannot push below with this approach
6. **Boundary identified**: remove_coef > 15 causes MMLU degradation without unlearning benefit

### Recommended Settings
- **For maximum unlearning**: retain_coef=1, remove_coef=3, 1 epoch (WMDP Bio: 29.84%)
- **For balanced performance**: retain_coef=5, remove_coef=5, 1 epoch (MMLU: 35.73%)
- **Avoid**: remove_coef > 15 (damages MMLU without improving unlearning)
- **Avoid longer training** - does not improve unlearning, wastes compute

## Notes
- All models saved to `models/EleutherAI/deep-ignorance-unfiltered_kl_ret{X}_rm{Y}[_ep4]`
- Training time: ~6 min/epoch with 4x GH200
- Evaluation time: ~15 minutes per model
