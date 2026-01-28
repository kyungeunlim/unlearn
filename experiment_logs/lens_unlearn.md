# Tuned Lens Unlearning Experiments

## Method
Unlearning via tuned lens activations - training model to match lens-projected (earlier layer) representations on forget data while maintaining performance on retain data.

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens (HuggingFace)
- **GPUs**: 4x GH200
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Baselines

| Model | WMDP Bio | MMLU STEM (deprecated) | MMLU |
|-------|----------|-----------|------|
| Target model (to unlearn) | 42.97% | 36.85% | ~43% |

## Results

### LoRA + AdamW Optimizer

| examples | epochs | steps | retain_coef | remove_coef | lr | batch | LoRA rank | retain_loss | forget_loss | WMDP Bio (↓) | MMLU STEM (deprecated) | MMLU | Notes |
|----------|--------|-------|-------------|-------------|-----|-------|-----------|-------------|-------------|--------------|-----------|------|-------|
| 8192     | 1      | 256   | 5.0         | 5.0         | 1e-3 | 32   | 16        | 1.19        | 1.04        | **23.16%**   | 34.57%    | **43.04%** | Best result |
| 16384    | 2      | 1024  | 5.0         | 5.0         | 1e-3 | 32   | 16        | 0.74        | 1.03        | 23.50%       | 23.88%    | -     | Over-unlearned, catastrophic MMLU loss |

### SFT + Muon Optimizer (torch.optim.Muon - PyTorch 2.10)

| examples | steps | retain_coef | remove_coef | muon_lr | adam_lr | WMDP Bio (↓) | MMLU STEM (deprecated) | MMLU | Notes |
|----------|-------|-------------|-------------|---------|---------|--------------|-----------|------|-------|
| 1024     | 32    | 1.0         | 2.0         | 1e-3    | 3e-4    | 24.67%       | 24.99%    | 28.71% | MMLU catastrophic |
| 1024     | 32    | 2.0         | 1.0         | 1e-3    | 3e-4    | 29.93%       | 31.34%    | 35.19% | Better balance |
| 1024     | 32    | 1.5         | 1.5         | 1e-3    | 3e-4    | 33.23%       | 30.99%    | **36.76%** | Best MMLU retention |
| 1024     | 32    | 3.0         | 1.0         | 1e-3    | 3e-4    | 27.42%       | 29.21%    | -     | Higher retain |
| 1024     | 32    | 2.5         | 0.5         | 1e-3    | 3e-4    | 42.42%       | 33.56%    | -     | Insufficient unlearn |

### SFT + AdamW Optimizer

| examples | steps | retain_coef | remove_coef | adam_lr | WMDP Bio (↓) | MMLU STEM (deprecated) | MMLU | Notes |
|----------|-------|-------------|-------------|---------|--------------|-----------|------|-------|
| 1024     | 32    | 5.0         | 2.0         | 1e-3    | 23.41%       | 24.64%    | 25.15% | MMLU catastrophic |
| 1024     | 32    | 20.0        | 2.0         | 1e-3    | 26.55%       | 26.20%    | 25.47% | |
| 1024     | 32    | 50.0        | 2.0         | 1e-3    | 24.74%       | 21.31%    | 22.97% | |
| 1024     | 32    | 200.0       | 2.0         | 1e-3    | 24.35%       | 21.69%    | - | |
| 1024     | 32    | 300.0       | 2.0         | 1e-3    | pending      | pending   | - | |
| 1024     | 32    | 500.0       | 2.0         | 1e-3    | 24.27%       | 28.45%    | - | Peak MMLU |
| 1024     | 32    | 700.0       | 2.0         | 1e-3    | pending      | pending   | - | |
| 1024     | 32    | 1000.0      | 2.0         | 1e-3    | 23.72%       | 21.25%    | - | MMLU drops again |

## Tampering Resistance (Finetune Attack)

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
