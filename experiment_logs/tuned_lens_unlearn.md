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
| 8192     | 1      | 256   | 5.0         | 5.0         | 1e-3 | 32   | 16        |             |             |              |             |           |        | Hook migration v1, eval failed (path), Job 2088655 |
| 8192     | 1      | 256   | 5.0         | 5.0         | 1e-3 | 32   | 16        |             |             |              | 24.65%      |           | 43.75% | Hook migration v2, Job 2088683 |
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

### SFT + AdamW Optimizer (SFT default lr=1e-5)

Runs below use a bf16 reference model (matching training precision), giving accurate retain_loss values. The retain_coeff schedule was also changed to start at 0.25x (ramping to 1.0x) instead of 0x, and lr warmup was added.

| examples | steps | retain_coef | remove_coef | adam_lr | warmup | retain_loss | forget_loss | WMDP Robust | MMLU | Notes |
|----------|-------|-------------|-------------|---------|--------|-------------|-------------|-------------|------|-------|
| -        | -     | -           | -           | -       | -      | -           | -           | 42.97%      | 45.10% | Baseline |
| 1024     | 32    | 500.0       | 1.0         | 1e-5    | 0.5    | 1.32        | 2.11        | 43.20%      | 45.14%    | Job 2075536: No unlearning, MMLU preserved |
| 1024     | 32    | 15.0        | 15.0        | 5e-4    | 0.5    | 892.15      | 1.91        | 26.38%      | 24.78%    | Job 2075550 |
| 1024     | 32    | 15.0        | 1.0         | 5e-4    | 0.5    | 586.26      | 2.24        | 27.76%      | 27.55%    | Job 2075554 |
| 1024     | 32    | 50.0        | 1.0         | 5e-4    | 0.5    | 552.87      | 2.33        | eval error  | 30.37%    | Job 2075558 |
| 1024     | 32    | 100.0       | 1.0         | 5e-4    | 0.5    | 585.73      | 1.56        | eval error  | 33.04%    | Job 2075559 |
| 1024     | 32    | 200.0       | 1.0         | 5e-4    | 0.5    | 557.40      | 2.28        | 32.60%      | 34.47%    | Job 2075571 |
| 1024     | 32    | 400.0       | 1.0         | 5e-4    | 0.5    | 6714.06     | 1.92        | 24.54%      | 24.41%    | Job 2075572: Unstable, model collapsed |
| 1024     | 32    | 400.0       | 5.0         | 1e-5    | 0.5    | 1.13        | 2.21        | 43.32%      | cancelled | Job 2075576: No unlearning |
| 1024     | 32    | 400.0       | 10.0        | 1e-5    | 0.5    | 1.29        | 2.21        | 43.20%      | cancelled | Job 2075577: No unlearning |
| 1024     | 32    | 400.0       | 20.0        | 1e-5    | 0.5    | 1.06        | 2.21        | 43.32%      | cancelled | Job 2075578: No unlearning |
| 1024     | 32    | 400.0       | 5.0         | 5e-5    | 0.5    | 19.49       | 2.12        | eval stuck  | eval stuck | Job 2075580 |
| 1024     | 32    | 400.0       | 5.0         | 1e-4    | 0.5    | 108.73      | 2.24        | 44.59%      | 45.98%    | Job 2075581: No unlearning |
| 1024     | 32    | 400.0       | 5.0         | 2e-4    | 0.5    | 190.69      | 2.26        | eval stuck  | eval stuck | Job 2075582 |
| 1024     | 32    | 400.0       | 5.0         | 3e-4    | 0.5    | 273.51      | 2.29        | 42.97%      | 43.80%    | Job 2075773: No unlearning |
| 1024     | 32    | 400.0       | 5.0         | 4e-4    | 0.5    | pending     | pending     | pending     | pending   | Job 2075784 |
| 1024     | 32    | 400.0       | 5.0         | 2e-4    | 0.5    | pending     | pending     | pending     | pending   | Job 2075785 |

### FSDP2 + Muon Optimizer (2-node, 8x GH200)

| examples | steps | retain_coef | remove_coef | muon_lr | adam_lr | retain_loss | forget_loss | WMDP Robust | MMLU | Notes |
|----------|-------|-------------|-------------|---------|---------|-------------|-------------|-------------|------|-------|
| -        | -     | -           | -           | -       | -       | -           | -           | 42.97%      | 45.10% | Baseline |
| 128      | 4     | 5.0         | 5.0         | 0.02    | 3e-4    | 0.00        | 2.23        | 26.73%      | 22.95% | Job 2075575: MMLU collapsed, retain warmup too slow |
| 128      | 4     | 50.0        | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075584 |
| 128      | 4     | 100.0       | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075585 |
| 128      | 4     | 400.0       | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075586 |
| 1024     | 32    | 20.0        | 5.0         | 0.02    | 3e-4    | 5.49        | -           | eval fail   | eval fail | Job 2075594: save failed (no FULL_STATE_DICT) |
| 1024     | 32    | 50.0        | 5.0         | 0.02    | 3e-4    | -0.73       | -           | eval fail   | eval fail | Job 2075595: save failed |
| 1024     | 32    | 100.0       | 5.0         | 0.02    | 3e-4    | -11.10      | -           | eval fail   | eval fail | Job 2075596: save failed |
| 1024     | 32    | 200.0       | 5.0         | 0.02    | 3e-4    | -31.84      | -           | eval fail   | eval fail | Job 2075597: save failed |
| 1024     | 32    | 20.0        | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075613 |
| 1024     | 32    | 50.0        | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075614 |
| 1024     | 32    | 100.0       | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075615 |
| 1024     | 32    | 200.0       | 5.0         | 0.02    | 3e-4    | pending     | pending     | pending     | pending | Job 2075616 |

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


## Archive

### Using a 4-bit reference model in SFT + AdamW

| examples | steps | retain_coef | remove_coef | adam_lr | retain_loss | forget_loss | WMDP Robust | MMLU | Notes |
|----------|-------|-------------|-------------|---------|-------------|-------------|-------------|------|-------|
| -        | -     | -           | -           | -       | -           | -           | 42.97%      | 45.10% | Baseline |
| 1024     | 32    | 5.0         | 5.0         | 1e-5    | 30.31       | 2.18        | 43.43%      | 45.09%    | Job 2045965 |
| 1024     | 32    | 5.0         | 5.0         | 1e-4    | 167.60      | 2.05        | 43.55%      | 42.42%    | Job 2046753 |
| examples | steps | retain_coef | remove_coef | adam_lr | retain_loss | forget_loss | WMDP Robust | MMLU | Notes |
|----------|-------|-------------|-------------|---------|-------------|-------------|-------------|------|-------|
| -        | -     | -           | -           | -       | -           | -           | 42.97%      | 45.10% | Baseline |
| 1024     | 32    | 20.0        | 5.0         | 1e-3    | 11571.27    | 1.48        | 24.08%      | 26.90%    | Job 2047279 |
| 1024     | 32    | 10.0        | 1.0         | 1e-3    | 11340.50    | 1.31        | 26.73%      | 22.95%    | Job 2062489 |
| 1024     | 32    | 15.0        | 1.0         | 1e-3    | 11373.78    | 1.45        | 25.69%      | 25.51%    | Job 2062376 |
| 1024     | 32    | 20.0        | 1.0         | 1e-3    | 11311.43    | 1.48        | 26.04%      | 25.57%    | Job 2062377 |
| 1024     | 32    | 30.0        | 1.0         | 1e-3    | 10913.43    | 1.48        | 27.19%      | 22.95%    | Job 2062378 |
| 1024     | 32    | 80.0        | 1.0         | 1e-3    | 11646.10    | 1.60        | 24.88%      | 23.59%    | Job 2075380 |
| 1024     | 32    | 100.0       | 1.0         | 1e-3    | 11763.40    | 1.85        | 23.50%      | 24.65%    | Job 2075390 |
| 1024     | 32    | 200.0       | 1.0         | 1e-3    | 12793.33    | 2.15        | 24.08%      | 26.89%    | Job 2075391 |
| 1024     | 32    | 500.0       | 1.0         | 1e-3    | 11277.07    | 1.42        | 26.50%      | 22.95%    | Job 2075383 |


**Note:** All runs above used a 4-bit quantized reference model for the KL retain loss, inflating retain_loss by ~48 (quantization noise floor). After this the retain_coeff schedule was also changed to start at 0.25x (ramping to 1.0x) instead of 0x, and lr warmup was added.

### Using a high learning rate in SFT + AdamW
