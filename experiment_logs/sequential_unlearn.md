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

## Experiments

### L2 Norm Retain Loss (LoRA)

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 10 | 5 | 128 | 0.3134 | 0.4479 | |
| 2 | 28→16 (step 4) | 20 | 5 | 128 | 0.3076 | 0.4454 | |
| 3 | 28→16 (step 4) | 30 | 5 | 128 | 0.2995 | 0.4458 | |
| 4 | 28→16 (step 4) | 40 | 5 | 128 | 0.2961 | 0.4410 | |
| 5 | 28→16 (step 4) | 50 | 5 | 128 | **0.2684** | **0.4437** | Below random (0.25) |
| 6 | 28→16 (step 4) | 50 | 5 | 1280 | 0.2535 | 0.2540 | 10x examples, MMLU collapsed |
| 7 | 28→16 (step 4) | 50 | 5 | 128 | 0.2535 | 0.2540 | Hook migration v1 (4-layer retain), Job 2088651 |
| 8 | 28→16 (step 4) | 50 | 5 | 128 | 0.3076 | 0.4488 | Hook migration v2 (all-layer retain), Job 2088695 |

### KL Retain Loss (LoRA)

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 50 | 5 | 128 | 0.2673 | 0.2295 | Collapsed |
| 2 | 28→16 (step 4) | 20 | 10 | 128 | 0.2673 | - | Collapsed |
| 3 | 28→16 (step 4) | 30 | 10 | 128 | 0.2673 | - | Collapsed |
| 4 | 28→16 (step 4) | 20 | 20 | 128 | 0.2673 | 0.2295 | Collapsed |
| 5 | 28→16 (step 4) | 10 | 20 | 128 | 0.2673 | - | Collapsed |
| 6 | 28→16 (step 4) | 10 | 50 | 128 | 0.2673 | 0.2295 | Collapsed |
| 7 | 28→16 (step 4) | 5 | 50 | 128 | 0.2730 | 0.4034 | Works |
| 8 | 28→16 (step 4) | 5 | 100 | 128 | **0.2788** | **0.4195** | Best KL |
| 9 | 28→16 (step 4) | 5 | 100 | 1280 | 0.2673 | 0.2295 | 10x examples |
| 10 | 28→16 (step 4) | 5 | 100 | 128 | 0.2673 | 0.2295 | Hook migration v1, collapsed, Job 2088652 |
| 11 | 28→16 (step 4) | 5 | 100 | 128 | 0.2558 | 0.3532 | Hook migration v2, Job 2088696 |

### Max Entropy KL Forget Loss, L2 Retain Loss (LoRA)

1024 examples, layers 31→8 (step 4), 4 GPUs, pdbs=4, grad_accum=2, global batch 32, 32 steps/layer, 192 total steps.

| Run | Layers | remove_coef | retain_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 10 | 5 | 192 | 2.55 | 1.05 | 0.2857 | 0.4067 | |

### Max Entropy KL Forget Loss, KL Retain Loss (SFT, FSDP)

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 5 | 20 | 1e-3 | 768 | ~7300 | 1.22 | 0.2419 | 0.2453 | Collapsed |
| 2 | 31→11 (step 4) | 5 | 80 | 1e-5 | 768 | 6.09 | 2.03 | 0.4309 | 0.4515 | No effect |
| 3 | 31→11 (step 4) | 5 | 80 | 4e-4 | 768 | 797 | 1.15 | 0.2661 | 0.2384 | Collapsed |
| 4 | 31→11 (step 4) | 5 | 200 | 1e-5 | 768 | 5.96 | 2.03 | 0.4332 | 0.4516 | No effect |
| 5 | 31→11 (step 4) | 5 | 200 | 4e-4 | 768 | 262 | 1.91 | 0.2696 | 0.2398 | Collapsed |
| 6 | 31→11 (step 4) | 5 | 400 | 1e-5 | 768 | 5.71 | 2.03 | 0.4332 | 0.4511 | No effect |
| 7 | 31→11 (step 4) | 5 | 400 | 4e-4 | 768 | 584 | 1.49 | 0.2327 | 0.2453 | Collapsed |
| 8 | 31→11 (step 4) | 5 | 2000 | 1e-5 | 768 | 3.13 | 1.92 | 0.4332 | 0.4516 | No effect |
| 9 | 31→11 (step 4) | 5 | 2000 | 4e-4 | 768 | 615 | 1.72 | 0.2650 | 0.2329 | Collapsed |
| 10 | 31→11 (step 4) | 5 | 80 | 5e-5 | 768 | 50.66 | 1.97 | 0.4355 | 0.4517 | No effect |
| 11 | 31→11 (step 4) | 5 | 200 | 5e-5 | 768 | 41.43 | 1.97 | 0.4366 | 0.4492 | No effect |
| 12 | 31→11 (step 4) | 5 | 400 | 5e-5 | 768 | 44.03 | 1.97 | 0.4286 | 0.4510 | No effect |
| 13 | 31→11 (step 4) | 5 | 2000 | 5e-5 | 768 | 38.69 | 1.97 | 0.4355 | 0.4507 | No effect |
| 14 | 31→11 (step 4) | 5 | 80 | 8e-5 | 768 | 60.59 | 1.97 | 0.4366 | 0.4432 | No effect |
| 15 | 31→11 (step 4) | 5 | 200 | 8e-5 | 768 | 63.20 | 1.97 | 0.4332 | 0.4426 | No effect |
| 16 | 31→11 (step 4) | 5 | 400 | 8e-5 | 768 | 60.81 | 1.97 | 0.4343 | 0.4454 | No effect |
| 17 | 31→11 (step 4) | 5 | 2000 | 8e-5 | 768 | 59.77 | 1.97 | 0.4274 | 0.4440 | No effect |
| 18 | 31→11 (step 4) | 5 | 80 | 1e-4 | 768 | 88.39 | 1.96 | 0.4251 | 0.4378 | No effect |
| 19 | 31→11 (step 4) | 5 | 200 | 1e-4 | 768 | 67.19 | 1.97 | 0.4251 | 0.4401 | No effect |
| 20 | 31→11 (step 4) | 5 | 400 | 1e-4 | 768 | 70.40 | 1.97 | 0.4286 | 0.4325 | No effect |
| 21 | 31→11 (step 4) | 5 | 2000 | 1e-4 | 768 | 73.30 | 1.98 | 0.4286 | 0.4363 | No effect |
| 22 | 31→11 (step 4) | 5 | 200 | 1.5e-4 | 768 | 98.78 | 1.95 | 0.4009 | 0.4241 | |
| 23 | 31→11 (step 4) | 5 | 2000 | 1.5e-4 | 768 | 93.86 | 1.98 | 0.4055 | 0.4139 | |
| 24 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 186.70 | 1.96 | 0.3629 | 0.3821 | |
| 25 | 31→11 (step 4) | 5 | 2000 | 2e-4 | 768 | 125.33 | 1.97 | 0.3675 | 0.3945 | |
| 26 | 31→11 (step 4) | 5 | 200 | 3e-4 | 768 | 165.14 | 1.92 | 0.3134 | 0.3186 | Collapsed |
| 27 | 31→11 (step 4) | 5 | 2000 | 3e-4 | 768 | 208.31 | 1.82 | 0.3065 | 0.3154 | Collapsed |

### Max Entropy KL Forget Loss, KL Retain Loss (SFT, FSDP, Same-Sign Grads)

Same-sign gradient filtering: separate backward passes for retain and forget losses. Only parameter elements where both gradients agree in sign receive updates. Combined with layer-wise gradient freezing.

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 5 | 200 | 2e-3 | 768 | | | | | |
| 2 | 31→11 (step 4) | 5 | 2000 | 2e-3 | 768 | | | | | |
| 3 | 31→11 (step 4) | 5 | 200 | 3e-3 | 768 | | | | | |
| 4 | 31→11 (step 4) | 5 | 2000 | 3e-3 | 768 | | | | | |
| 5 | 31→11 (step 4) | 5 | 200 | 5e-3 | 768 | | | | | |
| 6 | 31→11 (step 4) | 5 | 2000 | 5e-3 | 768 | | | | | |
| 7 | 31→11 (step 4) | 5 | 200 | 1e-2 | 768 | | | | | |
| 8 | 31→11 (step 4) | 5 | 2000 | 1e-2 | 768 | | | | | |

## Hyperparameters (all runs)

| Parameter | Value |
|-----------|-------|
| per_device_batch_size | 4 |
| gradient_accumulation | 2 |
| global_batch_size | 32 |
| learning_rate | 1e-3 |
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
