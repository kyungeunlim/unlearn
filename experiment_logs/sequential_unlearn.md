# Sequential Unlearning Experiments

Sequential back-to-front unlearning: for each layer L (from last to first), use the original model's layers L→N as a "probe" to compute forget loss while retaining performance on retain data by taking a retain loss through the updated model.

Note that hyperparameters don't transfer between number of training steps.

## Method

**Training**: LoRA fine-tuning (not full SFT)
- LoRA rank: 16
- LoRA alpha: 16
- Target modules: default (all linear layers)

**Losses**:
- Forget loss: Cross-entropy with random targets through original model's remaining layers (maximizes entropy)
- Retain loss: L2 norm between current and original hidden states across all layers

**Loss scheduling**:
- retain_coeff ramps from 0 → retain_coef over training
- forget_coeff ramps from remove_coef → 0.75*remove_coef over training

## Initial Experiments

### L2 Norm Retain Loss (LoRA)

Retain L2 computed at fixed layers [5,10,15,20,25,30], not at the current target layer. LoRA params in all layers receive gradient regardless, so retain gradients were non-zero, but the L2 constraint was on different layers than the forget target.

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 10 | 5 | 128 | 0.3134 | 0.4479 | |
| 2 | 28→16 (step 4) | 20 | 5 | 128 | 0.3076 | 0.4454 | |
| 3 | 28→16 (step 4) | 30 | 5 | 128 | 0.2995 | 0.4458 | |
| 4 | 28→16 (step 4) | 40 | 5 | 128 | 0.2961 | 0.4410 | |
| 5 | 28→16 (step 4) | 50 | 5 | 128 | **0.2684** | **0.4437** | Random (~0.25) |
| 6 | 28→16 (step 4) | 50 | 5 | 1280 | 0.2535 | 0.2540 | 10x examples, MMLU collapsed |
| 7 | 28→16 (step 4) | 50 | 5 | 128 | 0.2535 | 0.2540 | Hook migration v1 (4-layer retain), Job 2088651 |
| 8 | 28→16 (step 4) | 50 | 5 | 128 | 0.3076 | 0.4488 | Hook migration v2 (all-layer retain), Job 2088695 |

### KL Retain Loss (LoRA) - Worse Than L2 Norm Activation Retain

Retain KL computed on final logits. Gradients flow through all LoRA params including the target layer.

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 50 | 5 | 128 | 0.2673 | 0.2295 | Collapsed |
| 7 | 28→16 (step 4) | 5 | 50 | 128 | 0.2730 | 0.4034 | Works |
| 8 | 28→16 (step 4) | 5 | 100 | 128 | **0.2788** | **0.4195** | |
| 9 | 28→16 (step 4) | 5 | 100 | 1280 | 0.2673 | 0.2295 | 10x examples |
| 12 | 28→16 (step 4) | 5 | 100 | 128 | **0.3237** | **0.4375** | |

### Max Entropy KL Forget Loss, L2 Retain Loss (LoRA)

1024 examples, layers 31→8 (step 4), 4 GPUs, pdbs=4, grad_accum=2, global batch 32, 32 steps/layer, 192 total steps.

Retain L2 computed at fixed layers [5,10,15,20,25,30], not at the current target layer. LoRA params in all layers receive gradient regardless, so retain gradients were non-zero, but the L2 constraint was on different layers than the forget target.

| Run | Layers | remove_coef | retain_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 10 | 5 | 192 | 2.55 | 1.05 | **0.2857** | **0.4067** | |

### Max Entropy KL Forget Loss, L2 Retain Loss (SFT, some MMLU degradation, more unlearning)

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

Retain L2 computed at the current target layer's output. Only the target layer's parameters are unfrozen per phase.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 5 | 5 | 2e-4 | 768 | 1.28 | 1.93 | 0.3929 | 0.4462 | Job 2110034 |
| 2 | 31→11 (step 4) | 14 | 1 | 2e-4 | 768 | 1.70 | 1.61 | **0.2834** | **0.4239** | Job 2110068 |
| 3 | 31→1 (step 2) | 14 | 2 | 2e-4 | 2048 | | | | | Whole model |

### Max Entropy KL Forget Loss, KL Retain Loss (Naive SFT, breaks differential unlearning)

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

Retain KL computed on final logits. Gradients flow through all layers but only the target layer's grads are kept.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| 18 | 31→11 (step 4) | 5 | 80 | 1e-4 | 768 | 88.39 | 1.96 | 0.4251 | 0.4378 | No effect |
| 21 | 31→11 (step 4) | 5 | 2000 | 1e-4 | 768 | 73.30 | 1.98 | 0.4286 | 0.4363 | No effect |

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| 26 | 31→11 (step 4) | 5 | 200 | 3e-4 | 768 | 165.14 | 1.92 | 0.3134 | 0.3186 | Twin Degradation |
| 27 | 31→11 (step 4) | 5 | 2000 | 3e-4 | 768 | 208.31 | 1.82 | 0.3065 | 0.3154 | Twin Degradation |

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| 22 | 31→11 (step 4) | 5 | 200 | 1.5e-4 | 768 | 98.78 | 1.95 | 0.4009 | 0.4241 | Twin Degradation |
| 23 | 31→11 (step 4) | 5 | 2000 | 1.5e-4 | 768 | 93.86 | 1.98 | 0.4055 | 0.4139 | Twin Degradation |

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| 24 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 186.70 | 1.96 | 0.3629 | 0.3821 | Twin Degradation |
| 25 | 31→11 (step 4) | 5 | 2000 | 2e-4 | 768 | 125.33 | 1.97 | 0.3675 | 0.3945 | Twin Degradation |
| 28 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 8.57 | 2.03 | 0.4343 | 0.4528 | max_grad_norm=0 |

## Stabilizing SFT KL Retain Loss - Ablations

All ablations use max entropy KL forget loss, KL retain loss (on final logits, gradients flow through all layers but only target layer's grads kept), full SFT with same-sign gradient filtering and layer-wise gradient freezing. 1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

### Max Entropy KL Forget Loss, NLL Retain Loss (Potential unlearning, good retention)

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| 29 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 1.70 | 1.87 | **0.3986** | **0.4519** | nll_retain |

### Same-Sign Grads (Some Unlearning, Partial MMLU Degradation)

Separate backward passes for retain and forget losses. Only update parameter elements where both gradients agree in sign.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 5 | 20 | 2e-4 | 768 | 126.39 | 1.56 | 0.2949 | 0.3788 | |
| 2 | 31→11 (step 4) | 5 | 80 | 2e-4 | 768 | 126.52 | 1.56 | 0.2938 | 0.3787 | |
| 3 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 148.53 | 1.54 | 0.2938 | 0.3786 | |
| 4 | 31→11 (step 4) | 5 | 2000 | 2e-4 | 768 | 126.40 | 1.56 | 0.2938 | 0.3782 | |
| 29 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 148.45 | 1.54 | 0.2938 | 0.3793 | max_grad_norm=0.3 |

### Same-Sign + Asymmetric Filter (Little Unlearning)

Forget grads zeroed where they disagree with retain; all retain grads kept.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 5 | 31→11 (step 4) | 5 | 5 | 2e-4 | 768 | 9.47 | 1.97 | 0.4240 | | |
| 12 | 31→11 (step 4) | 5 | 2000 | 2e-4 | 768 | 10.93 | 1.97 | 0.4274 | | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero (Partial MMLU Degradation)

Retain coeff ramps from 0 (instead of 0.25x) over training.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 13 | 31→11 (step 4) | 5 | 1 | 5e-4 | 768 | 40.03 | 1.81 | 0.4101 | 0.4511 | |
| 14 | 31→11 (step 4) | 5 | 1 | 1e-3 | 768 | 74.63 | 1.76 | 0.3825 | 0.4383 | |
| 15 | 31→11 (step 4) | 5 | 1 | 2e-3 | 768 | 191.64 | 1.50 | 0.3560 | 0.3947 | |
| 16 | 31→11 (step 4) | 5 | 5 | 1e-3 | 768 | | | 0.4528 | | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero + L2-SP (Partial Unlearning + Good MMLU Retention)

L2-SP: regularize weights toward pretrained initialization (λ||w − w₀||²) computed via forward hook on the target layer.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 17 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | 74.28 | 1.76 | **0.3963** | **0.4593** | |
| 18 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.1 | 768 | 42.37 | 1.84 | 0.4136 | 0.4494 | |
| 19 | 31→11 (step 4) | 5 | 1 | 1e-3 | 1.0 | 768 | 18.49 | 1.96 | 0.4332 | 0.4517 | |
| 20 | 31→11 (step 4) | 5 | 1 | 2e-3 | 0.1 | 768 | 106.20 | 1.75 | 0.4009 | 0.4270 | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero + L2-SP + UltraChat

Same as above with UltraChat mixed into retain data.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 21 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | | | | | |

### Ramp Retain from Zero + L2-SP + UltraChat (No Same-Sign/Asymmetric)

L2-SP + ramp retain from zero + UltraChat, without gradient filtering.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 22 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | | | | | |

### L2-SP Only (No Same-Sign, Asymmetric, or Ramp)

L2-SP regularization without gradient filtering. l2sp_coef=0.01.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 28 | 31→11 (step 4) | 40 | 200 | 2e-4 | 0.01 | 768 | 10.64 | 1.92 | 0.4297 | 0.4497 | No effect |
| 32 | 31→11 (step 4) | 40 | 5 | 1e-4 | 0.01 | 768 | 3.77 | 1.88 | 0.4320 | 0.4506 | No effect |
| 37 | 31→11 (step 4) | 20 | 5 | 1e-3 | 0.01 | 768 | 81.46 | 1.94 | 0.4055 | 0.4194 | |
| 38 | 31→11 (step 4) | 2 | 30 | 1e-3 | 0.01 | 768 | | | | | |

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| per_device_batch_size | 4 |
| gradient_accumulation | 2 |
| global_batch_size | 32 |
| lora_r | 16 |
| epochs_per_layer | 1 |
| layers_unlearned | 28, 24, 20, 16 |
| optimizer | AdamW |
| bf16 | True |
| gradient_checkpointing | True |

## Tampering Attack Results

Fine-tuning unlearned models on WMDP-Bio-Remove dataset to test recovery:

| Model | Step 0 | Step 10 | Step 100 |
|-------|--------|---------|----------|
| Baseline (target) | 0.4297 | - | - |
| L2 (rm50/ret5) | 0.2696 | 0.4055 | 0.4182 |
| KL (rm5/ret100) | 0.2788 | 0.4078 | 0.4217 |

Both models show rapid recovery (~10 steps).
Plots saved in `runs/tamper_attack/` and `runs/tamper_kl_rm5_ret100/`.
