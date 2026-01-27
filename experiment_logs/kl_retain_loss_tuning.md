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

| retain_coef | remove_coef | WMDP Bio (↓ better) | MMLU (↑ better) | Status |
|-------------|-------------|---------------------|-----------------|--------|
| 1           | 3           | 29.84% ± 1.55%      | 33.61% ± 0.40%  | ✓ |
| 3           | 5           | —                   | —               | not evaluated |
| 5           | 5           | 31.68% ± 1.58%      | 35.73% ± 0.40%  | ✓ |
| 10          | 5           | 30.76% ± 1.57%      | 34.98% ± 0.40%  | ✓ |
| 5           | 10          | —                   | —               | not evaluated |
| 10          | 10          | —                   | —               | not evaluated |

## Key Findings

### Boundary Analysis
- **WMDP Bio random chance**: ~25%
- **Achieved WMDP Bio**: 29.84% - 31.68% (incomplete unlearning, ~5-7% above random)
- **MMLU retention**: 33.61% - 35.73%

### Trade-offs
| Setting | WMDP Bio (unlearning) | MMLU (retention) | Notes |
|---------|----------------------|------------------|-------|
| Low retain (1,3) | Best (29.84%) | Worst (33.61%) | More aggressive unlearning hurts retention |
| Default (5,5) | Middle (31.68%) | Best (35.73%) | Good balance |
| High retain (10,5) | Middle (30.76%) | Good (34.98%) | High retain pressure doesn't improve MMLU much |

### Conclusions
1. **KL divergence retain loss works** but unlearning is incomplete (WMDP still 5-7% above random)
2. **retain_coef=5** appears optimal - best MMLU (35.73%) with reasonable WMDP (31.68%)
3. **Lower retain_coef** improves unlearning (29.84%) but damages general capability
4. **remove_coef** has minimal impact on unlearning effectiveness in this range

### Recommended Settings
- **For maximum unlearning**: retain_coef=1, remove_coef=3 (WMDP: 29.84%, MMLU: 33.61%)
- **For balanced performance**: retain_coef=5, remove_coef=5 (WMDP: 31.68%, MMLU: 35.73%)

## Notes
- All models saved to `models/EleutherAI/deep-ignorance-unfiltered_kl_ret{X}_rm{Y}`
- Training time per config: ~6 minutes with 4x GH200
- Evaluation time per config: ~15 minutes
