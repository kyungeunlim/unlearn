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

## Hyperparameters to Tune
- `retain_coef`: Coefficient for KL divergence retain loss (default: 5)
- `remove_coef`: Coefficient for circuit breaker loss (default: 5)

## Training Metrics (Step 32)

| retain_coef | remove_coef | retain_kl_loss | cb_loss | retain_acc | forget_acc |
|-------------|-------------|----------------|---------|------------|------------|
| 1           | 3           | 0.247          | 6.30    | 76.1%      | 71.4%      |
| 3           | 5           | 0.203          | 6.30    | 78.6%      | 71.5%      |
| 5           | 5           | 0.156          | 6.31    | 81.1%      | 71.5%      |
| 10          | 5           | 0.101          | 6.33    | 84.6%      | 71.6%      |
| 5           | 10          | 0.229          | 6.35    | 77.3%      | 71.1%      |
| 10          | 10          | 0.169          | 6.36    | 80.4%      | 71.1%      |

### Training Observations
- Higher `retain_coef` → lower KL loss, higher retain argmax accuracy
- `remove_coef` has minimal impact on forget_acc (~71% across all configs)
- Circuit breaker loss stable around 6.30-6.36

## Evaluation Results

| retain_coef | remove_coef | WMDP Bio (↓ better) | MMLU STEM (↑ better) | Status |
|-------------|-------------|---------------------|----------------------|--------|
| 1           | 3           | pending             | pending              | eval pending |
| 3           | 5           | pending             | pending              | eval pending |
| 5           | 5           | pending             | pending              | eval pending |
| 10          | 5           | pending             | pending              | eval pending |
| 5           | 10          | pending             | pending              | eval pending |
| 10          | 10          | pending             | pending              | eval pending |

## Notes
- WMDP Bio random chance = ~25%
- Target: WMDP Bio near 25%, MMLU STEM as high as possible
- All models saved to `models/EleutherAI/deep-ignorance-unfiltered_kl_ret{X}_rm{Y}`
