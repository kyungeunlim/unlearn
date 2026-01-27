# Tuned Lens Unlearning Experiments

## Method
Unlearning via tuned lens activations - training model to match lens-projected (earlier layer) representations on forget data while maintaining performance on retain data.

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens (HuggingFace)
- **GPUs**: 4x GH200
- **LoRA rank**: 16
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Baselines

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |

## Results

### AdamW Optimizer

| examples | epochs | steps | retain_coef | remove_coef | lr | batch | retain_loss | forget_loss | WMDP Bio (↓) | MMLU (↑) | Notes |
|----------|--------|-------|-------------|-------------|-----|-------|-------------|-------------|--------------|----------|-------|
| 8192     | 1      | 256   | 5.0         | 5.0         | 1e-3 | 32   | 1.19        | 1.04        | **23.16%**   | 34.35%   | Best result |
| 16384    | 2      | 1024  | 5.0         | 5.0         | 1e-3 | 32   | pending     | pending     | pending      | pending  | Long run |

### Muon Optimizer (Pre-Fix - retain_loss bug)

| examples | steps | retain_coef | remove_coef | muon_lr | adam_lr | WMDP Bio (↓) | MMLU (↑) | Notes |
|----------|-------|-------------|-------------|---------|---------|--------------|----------|-------|
| 1024     | 128   | 0.0         | 5.0         | 0.02    | 3e-4    | 24.74%       | 23.88%   | Over-unlearned |
| 1024     | 128   | 0.0         | 2.0         | 0.02    | 3e-4    | 24.74%       | 23.88%   | Same |
| 512      | 64    | 0.0         | 2.0         | 0.02    | 3e-4    | 24.74%       | 23.88%   | Same |
| 1024     | 128   | 0.0         | 2.0         | 0.005   | 3e-4    | 24.74%       | 23.88%   | Same |

### Muon Optimizer (torch.optim.Muon - PyTorch 2.10)

| examples | steps | retain_coef | remove_coef | muon_lr | adam_lr | WMDP Bio (↓) | MMLU (↑) | Notes |
|----------|-------|-------------|-------------|---------|---------|--------------|----------|-------|
| 1024     | 32    | 1.0         | 2.0         | 1e-3    | 3e-4    | 24.67%       | 24.99%   | MMLU catastrophic |

## Tampering Resistance (Finetune Attack)

Model: `deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0`

| Attack Config | Step 0 | Step 50 | Step 100 | Recovery |
|---------------|--------|---------|----------|----------|
| 512 examples, lr=2e-5, 1 epoch | 24.82% | 52.08% | 51.61% | +27% |

## Key Findings

1. **Best unlearning**: 8192 examples, AdamW → WMDP Bio 23.16% (below random), MMLU 34.35%
2. **Muon optimizer**: Tends to over-unlearn, damaging MMLU significantly
3. **Tampering vulnerability**: Finetuning on 512 bio examples recovers WMDP knowledge from 25% → 52% in 50 steps
4. **Trade-off**: Lower WMDP Bio correlates with lower MMLU retention

## Model Paths
- Best model: `models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0`
