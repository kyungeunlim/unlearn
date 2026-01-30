# Tuned Lens Unlearning Experiments

## Method
Unlearning via tuned lens activations - on forget data, representations at each layer are mapped to predictions using tuned lenses, then those predictions are unlearned using cross-entropy against random token labels (maximizing entropy). Retain loss is L2 norm between hidden states of the LoRA-adapted model and the frozen original model on retain data, preserving the original model's intermediate representations.

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens (HuggingFace)
- **GPUs**: 4x GH200
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Results

### LoRA + AdamW Optimizer

| examples | epochs | steps | retain_coef | remove_coef | lr | batch | LoRA rank | retain_loss | forget_loss | WMDP Bio (↓) | WMDP Robust | MMLU STEM (deprecated) | MMLU | Notes |
|----------|--------|-------|-------------|-------------|-----|-------|-----------|-------------|-------------|--------------|-------------|-----------|------|-------|
| -        | -      | -     | -           | -           | -   | -     | -         | -           | -           | 42.97%       | 42.97%      | 36.85%    | 45.10% | Baseline |
| 8192     | 1      | 256   | 5.0         | 5.0         | 1e-3 | 32   | 16        | 1.19        | 1.04        | **23.16%**   | 23.16%      | 34.57%    | **43.04%** | Best result |
| 16384    | 2      | 1024  | 5.0         | 5.0         | 1e-3 | 32   | 16        | 0.74        | 1.03        | 23.50%       | -           | 23.88%    | -     | Over-unlearned, catastrophic MMLU loss |

### SFT + Muon Optimizer (torch.optim.Muon - PyTorch 2.10)

| examples | steps | retain_coef | remove_coef | muon_lr | adam_lr | WMDP Bio (↓) | WMDP Robust | MMLU STEM (deprecated) | MMLU | Notes |
|----------|-------|-------------|-------------|---------|---------|--------------|-------------|-----------|------|-------|
| -        | -     | -           | -           | -       | -       | 42.97%       | 42.97%      | 36.85%    | 45.10% | Baseline |
| 1024     | 32    | 1.0         | 2.0         | 1e-3    | 3e-4    | 24.67%       | -           | 24.99%    | 28.71% | MMLU catastrophic |
| 1024     | 32    | 2.0         | 1.0         | 1e-3    | 3e-4    | 29.93%       | -           | 31.34%    | 35.19% | Better balance |
| 1024     | 32    | 1.5         | 1.5         | 1e-3    | 3e-4    | 33.23%       | 30.53%      | 30.99%    | **36.76%** | Best MMLU retention |
| 1024     | 32    | 3.0         | 1.0         | 1e-3    | 3e-4    | 27.42%       | -           | 29.21%    | -     | Higher retain |
| 1024     | 32    | 2.5         | 0.5         | 1e-3    | 3e-4    | 42.42%       | -           | 33.56%    | -     | Insufficient unlearn |

### SFT + AdamW Optimizer

| examples | steps | retain_coef | remove_coef | adam_lr | WMDP Robust | MMLU | Notes |
|----------|-------|-------------|-------------|---------|-------------|-----------|-------|
| -        | -     | -           | -           | -       | 42.97%      | 45.10% | Baseline |
| 1024     | 32    | 0.0         | 5.0         | 1e-5    | 42.86%      | timeout   | Job 2042124: No unlearning (baseline=42.97%), MMLU eval timed out |
| 1024     | 32    | 0.0         | 5.0         | 1e-5    | 42.17%      | 44.14%    | Job 2042919: SFT no retain (insufficient lr) |
| 1024     | 32    | 5.0         | 5.0         | 1e-5    | 43.43%      | 45.09%    | Job 2045965: No unlearning (lr too low), MMLU preserved |
| 1024     | 32    | 5.0         | 5.0         | 1e-4    | 43.55%      | 42.42%    | Job 2046753: Still no unlearning, MMLU slight degradation |
| 1024     | 32    | 20.0        | 5.0         | 1e-3    | 24.08%      | 26.90%    | Job 2047279: Strong unlearning but MMLU damaged |
| 1024     | 32    | 10.0        | 1.0         | 1e-3    | 26.73%      | 22.95%    | Job 2062489 |
| 1024     | 32    | 15.0        | 1.0         | 1e-3    | 25.69%      | 25.51%    | Job 2062376 |
| 1024     | 32    | 20.0        | 1.0         | 1e-3    | 26.04%      | 25.57%    | Job 2062377 |
| 1024     | 32    | 30.0        | 1.0         | 1e-3    | 27.19%      | 22.95%    | Job 2062378 |
| 1024     | 32    | 80.0        | 1.0         | 1e-3    | 24.88%      | 23.59%    | Job 2075380 |
| 1024     | 32    | 100.0       | 1.0         | 1e-3    | 23.50%      | 24.65%    | Job 2075390 |
| 1024     | 32    | 200.0       | 1.0         | 1e-3    | 24.08%      | 26.89%    | Job 2075391 |
| 1024     | 32    | 500.0       | 1.0         | 1e-3    | 26.50%      | 22.95%    | Job 2075383 |

## Tamper Resistance (Finetune Attack)

Model: `deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0`
Dataset: `Unlearning/WMDP-Bio-Remove-Dataset`
Batch size: 16

| Attack Config | Step 0 | Step 50 | Step 100 | Recovery |
|---------------|--------|---------|----------|----------|
| 512 examples, lr=2e-5, 1 epoch | 24.82% | 52.08% | 51.61% | +27% |

## Example Command
```bash
python unlearn/reference/cas/finetune_attack.py \
  --model_name models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0 \
  --num_train_examples 512 \
  --eval_every 10 \
  --epochs 1
```

## Key Findings

1. **Best unlearning**: 8192 examples, LoRA + AdamW → WMDP Bio 23.16% (below random), MMLU 43.04% (baseline retention)
2. **Longer training hurts**: 16384 examples (2 epochs) caused catastrophic MMLU loss (23.88% STEM) without improving unlearning
3. **Muon optimizer**: Tends to over-unlearn, damaging MMLU significantly
4. **Tampering vulnerability**: Finetuning on 512 bio examples recovers WMDP knowledge from 25% → 52% in 50 steps
5. **Trade-off**: Lower WMDP Bio correlates with lower MMLU retention
6. **SFT vs LoRA**: Full SFT with AdamW cannot achieve the same results as LoRA. Even with very high retain_coef (50.0), SFT causes catastrophic MMLU damage (21-26% STEM), while LoRA preserves MMLU at 43%. Full parameter updates are too aggressive for this unlearning task.

## Model Paths
- Best model: `models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0`
