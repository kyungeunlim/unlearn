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

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | correct_prob | mean_srank | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|--------------|------------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | Baseline |
| 1 | 28→16 (step 4) | 10 | 5 | 128 | 0.3134 | 0.4479 | | | |
| 2 | 28→16 (step 4) | 20 | 5 | 128 | 0.3076 | 0.4454 | | | |
| 3 | 28→16 (step 4) | 30 | 5 | 128 | 0.2995 | 0.4458 | | | |
| 4 | 28→16 (step 4) | 40 | 5 | 128 | 0.2961 | 0.4410 | | | |
| 5 | 28→16 (step 4) | 50 | 5 | 128 | **0.2684** | **0.4437** | 0.1355 | 1.30 | Random (~0.25) |
| 6 | 28→16 (step 4) | 50 | 5 | 1280 | 0.2535 | 0.2540 | | | 10x examples, MMLU collapsed |
| 8 | 28→16 (step 4) | 50 | 5 | 128 | 0.3076 | 0.4488 | | | Hook migration v2 (all-layer retain), Job 2088695 |
| 9 | 28→16 (step 4) | 50 | 5 | 128 | 0.3191 | 0.4500 | | | +ultrachat |
| 10 | 28→16 (step 4) | 50 | 5 | 128 | 0.2823 | 0.4303 | | | Replication of run 5, Job 2110179 |
| 11 | 28→16 (step 4) | 50 | 2 | 128 | 0.3088 | 0.4516 | | | +ultrachat, Job 2110779 |
| 12 | 28→16 (step 4) | 70 | 2 | 128 | 0.3053 | 0.4518 | | | +ultrachat, Job 2113085 |

### KL Retain Loss (LoRA) - Worse Than L2 Norm Activation Retain

Retain KL computed on final logits. Gradients flow through all LoRA params including the target layer.

The first successful run doesn't replicate - it may have been more effective due to a bug where different layers were targeted for retention.

Using relatively more retain layers then forget layers is an interesting strategy.

Even using different forget and retain layers is interesting. This could help circuits to be "rebuilt" rather than "modified".

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | correct_prob | mean_srank | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|--------------|------------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | Baseline |
| 1 | 28→16 (step 4) | 50 | 5 | 128 | 0.2673 | 0.2295 | | | Collapsed |
| 7 | 28→16 (step 4) | 5 | 50 | 128 | 0.2730 | 0.4034 | | | Works |
| 8 | 28→16 (step 4) | 5 | 100 | 128 | **0.2788** | **0.4195** | 0.1384 | 1.15 | I think this one might have targeted different layers for retain |
| 9 | 28→16 (step 4) | 5 | 100 | 1280 | 0.2673 | 0.2295 | | | 10x examples |
| 12 | 28→16 (step 4) | 5 | 100 | 128 | **0.3237** | **0.4375** | 0.1918 | 1.15 | |

### Max Entropy KL Forget Loss, L2 Retain Loss (LoRA)

1024 examples, layers 31→8 (step 4), 4 GPUs, pdbs=4, grad_accum=2, global batch 32, 32 steps/layer, 192 total steps.

Retain L2 computed at fixed layers [5,10,15,20,25,30], not at the current target layer. LoRA params in all layers receive gradient regardless, so retain gradients were non-zero, but the L2 constraint was on different layers than the forget target.

| Run | Layers | remove_coef | retain_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | correct_prob | mean_srank | Notes |
|-----|--------|-------------|-------------|-------|-------------|-------------|-----------------|------|--------------|------------|-------|
| - | - | - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | Baseline |
| 1 | 31→11 (step 4) | 10 | 5 | 192 | 2.55 | 1.05 | **0.2857** | **0.4067** | 0.1851 | 1.51 | |

### Max Entropy KL Forget Loss, L2 Retain Loss (SFT, some MMLU degradation, more unlearning)

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

Retain L2 computed at the current target layer's output. Only the target layer's parameters are unfrozen per phase.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | correct_prob | mean_srank | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|--------------|------------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | Baseline |
| 1 | 31→11 (step 4) | 5 | 5 | 2e-4 | 768 | 1.28 | 1.93 | 0.3929 | 0.4462 | | | Job 2110034 |
| 2 | 31→11 (step 4) | 14 | 1 | 2e-4 | 768 | 1.70 | 1.61 | **0.2834** | **0.4239** | 0.1394 | 3.09 | Job 2110068 |
| 3 | 31→1 (step 2) | 14 | 2 | 2e-4 | 2048 | | | | | | | Job 2110099, OOM at step ~1024 |
| 4 | 31→1 (step 2) | 10 | 2 | 2e-4 | 2048 | 0.15 | 1.94 | 0.2730 | 0.3618 | | | Job 2110783 |
| 5 | 31→1 (step 2) | 5 | 2 | 2e-4 | 2048 | 0.15 | 1.94 | 0.3065 | 0.4034 | | | Job 2110784 |
| 6 | 31→1 (step 2) | 2 | 2 | 2e-4 | 2048 | 0.14 | 1.94 | 0.3514 | 0.4328 | | | Job 2110785 |
| 7 | 31→1 (step 2) | 1 | 2 | 2e-4 | 2048 | 0.14 | 2.00 | 0.3917 | 0.4447 | | | Job 2110786 |
| 8 | 31→11 (step 4) | 14 | 1 | 2e-4 | 768 | 5.53 | 1.61 | **0.2949** | **0.4199** | 0.1410 | 2.65 | +ultrachat |

### Max Entropy KL Forget Loss, L2 Retain Loss (SFT + Muon Optimizer)

Same setup as above but using MuonAdamW optimizer (Muon for 2D hidden matrices, AdamW for embeddings/biases/norms). muon_lr=0.02, adam_lr=2e-4. +ultrachat.

| Run | Layers | remove_coef | retain_coef | lr (muon) | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | correct_prob | mean_erank | mean_srank | Notes |
|-----|--------|-------------|-------------|-----------|-------|-------------|-------------|-----------------|------|--------------|------------|------------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | | Baseline |
| 1 | 31→11 (step 4) | 14 | 1 | 0.02 | 768 | ~6.8 | 1.04 | **0.2707** | **0.4160** | 0.1418 | 2189 | 3.15 | Job 2110810 |
| 2 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.9 | 1.07 | **0.3007** | **0.4367** | 0.1577 | 2626 | 3.28 | Job 2110867 |
| 3 | 31→11 (step 4) | 5 | 2 | 0.02 | 1536 | ~3.5 | 1.05 | 0.3479 | 0.4410 | | | | Job 2113120, epochs_per_layer=2 |
| 4 | 31→11 (step 4) | 5 | 2 | 1e-3 | 768 | ~5.9 | 1.07 | 0.3018 | 0.4368 | | | | Job 2113524 |
| 5 | 31→11 (step 4) | 14 | 1 | 1e-3 | 768 | ~6.8 | 1.04 | 0.2707 | 0.4160 | | | | Job 2113525 |
| 6 | 31→11 (step 4) | 5 | 2 | 2e-4 | 768 | ~5.9 | 1.07 | 0.3007 | 0.4368 | | | | Job 2113528 |
| 7 | 31→11 (step 4) | 14 | 1 | 2e-4 | 768 | ~6.8 | 1.04 | 0.2707 | 0.4160 | | | | Job 2113529 |
| 8 | 31→11 (step 4) | 5 | 16 | 0.02 | 3072 | ~1.68 | ~1.08 | 0.3813 | 0.4457 | | | | Job 2113545, epochs_per_layer=4 |
| 9 | 31→11 (step 4) | 5 | 20 | 0.02 | 3072 | ~1.84 | ~1.19 | 0.4032 | 0.4451 | | | | Job 2113546, epochs_per_layer=4 |
| 10 | 31→11 (step 4) | 5 | 30 | 0.02 | 3072 | ~1.87 | ~1.72 | 0.4401 | 0.4484 | | | | Job 2113547, epochs_per_layer=4 |
| 11 | 31→11 (step 4) | 5 | 40 | 0.02 | 3072 | ~1.89 | ~1.84 | 0.4470 | 0.4492 | | | | Job 2113548, epochs_per_layer=4 |
| 12 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.7 | ~1.04 | **0.2903** | **0.4467** | 0.1573 | 2417 | 3.23 | Job 2125986, +regex keyword_mask forget loss |
| 13 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.3 | ~1.03 | 0.3065 | 0.4432 | | 3287 | | Job 2126016, +activation_mask (layer 16, threshold 0.63, ~19.4% tokens) |
| 14 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.2 | ~1.03 | 0.3502 | 0.4457 | | | | Job 2126061, +sae_mask (11 WMDP latents, 0.92% tokens) |
| 15 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.2 | ~1.03 | 0.3237 | 0.4415 | | | | Job 2126084, +sae_mask (1149 bio latents, 39% tokens) |
| 16 | 31→0 (step 1) | 5 | 2 | 0.02 | 1024 | | ~1.02 | 0.3353 | 0.4443 | | | | Job 2133104, +regex keyword_mask, all layers, 32 steps/layer |
| 16c | 31→0 (step 1) | 5 | 2 | 0.02 | 4096 | | ~1.02 | **0.2995** | **0.4469** | | | | Job 2133383, +regex keyword_mask, all layers, 128 steps/layer |
| 16d | 31→0 (step 1) | 5 | 2 | 0.02 | 1024 | | ~1.03 | 0.3341 | 0.4166 | | | | Job 2133391, +regex keyword_mask, all layers, 32 steps/layer, +maintain_unlearned |
| 17 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.3 | ~1.03 | 0.3525 | 0.4420 | | | | Job 2127358, +probe_mask (layer 11 linear, 10.5% tokens) |
| 18 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | ~5.9 | ~1.03 | 0.3848 | 0.4499 | 0.1631 | 2999 | 18.81 | Job 2133026, +regex keyword_mask, +l2sp_coef=0.01 |

### Top-K Entropy Forget Loss (K=100), L2 Retain Loss (LoRA)

Maximizes entropy over only the top-100 most likely tokens at each position (normalized by log(K) instead of log(V)).

1024 examples, layers 28→16 (step 4), 4 GPUs, pdbs=4, grad_accum=2, global batch 32, 32 steps/layer, 128 total steps.

| Run | Layers | remove_coef | retain_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 50 | 5 | 128 | ~2.74 | ~1.00 | **0.2857** | **0.4499** | Job 2133393 |
| 2 | 31→0 (step 1) | 50 | 5 | 1024 | | ~1.00 | 0.2431 | 0.2459 | Job 2133442, all layers, 32 steps/layer, MMLU collapsed |
| 5 | 31→0 (step 1) | 5 | 5 | 1024 | ~2.96 | ~1.07 | 0.2546 | 0.2450 | Job 2140894, all layers, 32 steps/layer, MMLU collapsed |
| 6 | 31→0 (step 1) | 10 | 5 | 1024 | ~3.17 | ~1.26 | 0.2350 | 0.2465 | Job 2140895, all layers, 32 steps/layer, MMLU collapsed |
| 7 | 31→0 (step 1) | 20 | 5 | 1024 | ~3.21 | ~1.18 | 0.2465 | 0.3863 | Job 2140896, all layers, 32 steps/layer |
| 8 | 31→0 (step 1) | 12 | 5 | 1024 | | | | | Job 2208067, all layers, 32 steps/layer |
| 9 | 31→0 (step 1) | 15 | 5 | 1024 | | | | | Job 2208068, all layers, 32 steps/layer |

### Top-K Entropy Forget Loss, L2 Retain Loss (SFT + Muon + Keyword Mask)

1024 examples, layers 31→11 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps. +regex keyword_mask, +ultrachat.

| Run | K | Layers | remove_coef | retain_coef | lr (muon) | Steps | Batch | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|---|--------|-------------|-------------|-----------|-------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 100 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | 32 | ~2.5 | ~1.01 | **0.3260** | **0.4380** | Job 2133394 |
| 2 | 10 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | 32 | ~5.88 | ~1.01 | **0.3122** | **0.4435** | Job 2133435 |
| 3 | 50 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | 32 | 4.77 | 1.01 | 0.3433 | 0.4379 | Job 2133436 |
| 4 | 200 | 31→11 (step 4) | 5 | 2 | 0.02 | 768 | 32 | 4.78 | 1.01 | 0.3353 | 0.4390 | Job 2133437 |
| 5 | 100 | 31→11 (step 4) | 5 | 2 | 2e-4 | 3072 | 32 | | | | | Job 2209376, epochs_per_layer=4 |

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

### Max Entropy KL Forget Loss, NLL Retain Loss (Weak unlearning, good retention)

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | correct_prob | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|--------------|-------|
| 29 | 31→11 (step 4) | 5 | 200 | 2e-4 | 768 | 1.70 | 1.87 | 0.3986 | 0.4519 | 0.2309 | |
| 39 | 31→1 (step 2) | 5 | 200 | 2e-4 | 2048 | 0.97 | 1.97 | 0.3825 | 0.4387 | | |
| 41 | 31→1 (step 2) | 10 | 200 | 2e-4 | 2048 | 1.00 | 2.00 | 0.3733 | 0.4239 | | |
| 42 | 31→1 (step 2) | 10 | 500 | 2e-4 | 2048 | 1.00 | 1.94 | 0.3859 | 0.4385 | | |
| 43 | 31→1 (step 2) | 5 | 500 | 2e-4 | 2048 | 1.01 | 1.94 | 0.4021 | 0.4410 | | |
| 44 | 31→1 (step 2) | 2 | 200 | 2e-4 | 2048 | 0.95 | 2.00 | 0.3975 | 0.4412 | | |

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

### Same-Sign + Asymmetric Filter (Little Unlearning, needs more tuning)

Forget grads zeroed where they disagree with retain; all retain grads kept.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 5 | 31→11 (step 4) | 5 | 5 | 2e-4 | 768 | 9.47 | 1.97 | 0.4240 | | |
| 12 | 31→11 (step 4) | 5 | 2000 | 2e-4 | 768 | 10.93 | 1.97 | 0.4274 | | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero (MMLU Degradation)

Retain coeff ramps from 0 (instead of 0.25x) over training.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 13 | 31→11 (step 4) | 5 | 1 | 5e-4 | 768 | 40.03 | 1.81 | 0.4101 | 0.4511 | |
| 14 | 31→11 (step 4) | 5 | 1 | 1e-3 | 768 | 74.63 | 1.76 | 0.3825 | 0.4383 | |
| 15 | 31→11 (step 4) | 5 | 1 | 2e-3 | 768 | 191.64 | 1.50 | 0.3560 | 0.3947 | |
| 16 | 31→11 (step 4) | 5 | 5 | 1e-3 | 768 | | | 0.4528 | | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero + L2-SP (Weak Unlearning + Good MMLU Retention)

L2-SP: regularize weights toward pretrained initialization (λ||w − w₀||²) computed via forward hook on the target layer.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | correct_prob | mean_erank | mean_srank | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|--------------|------------|------------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | 0.2008 | | | Baseline |
| 17 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | 74.28 | 1.76 | 0.3963 | 0.4593 | 0.1964 | 2986 | 18.26 | |
| 18 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.1 | 768 | 42.37 | 1.84 | 0.4136 | 0.4494 | | | | |
| 19 | 31→11 (step 4) | 5 | 1 | 1e-3 | 1.0 | 768 | 18.49 | 1.96 | 0.4332 | 0.4517 | | | | |
| 20 | 31→11 (step 4) | 5 | 1 | 2e-3 | 0.1 | 768 | 106.20 | 1.75 | 0.4009 | 0.4270 | | | | |

### Same-Sign + Asymmetric Filter + Ramp Retain from Zero + L2-SP + UltraChat (Undertuned, no change)

Same as above with UltraChat mixed into retain data.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 21 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | 32.0 | 1.74 | 0.4147 | 0.4418 | Job 2110097 |

### Ramp Retain from Zero + L2-SP + UltraChat (No Same-Sign/Asymmetric)

L2-SP + ramp retain from zero + UltraChat, without gradient filtering.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 22 | 31→11 (step 4) | 5 | 1 | 1e-3 | 0.01 | 768 | 32.0 | 1.48 | 0.3998 | 0.4326 | Job 2110098 |

### L2-SP Only (No Same-Sign, Asymmetric, or Ramp)

L2-SP regularization without gradient filtering. l2sp_coef=0.01.

| Run | Layers | remove_coef | retain_coef | lr | l2sp_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 28 | 31→11 (step 4) | 40 | 200 | 2e-4 | 0.01 | 768 | 10.64 | 1.92 | 0.4297 | 0.4497 | No effect |
| 32 | 31→11 (step 4) | 40 | 5 | 1e-4 | 0.01 | 768 | 3.77 | 1.88 | 0.4320 | 0.4506 | No effect |
| 37 | 31→11 (step 4) | 20 | 5 | 1e-3 | 0.01 | 768 | 81.46 | 1.94 | 0.4055 | 0.4194 | |
| 38 | 31→11 (step 4) | 2 | 30 | 1e-3 | 0.01 | 768 | 930.06 | 1.95 | 0.4274 | 0.4588 | No effect |
| 40 | 31→11 (step 4) | 2 | 30 | 1e-3 | 0.01 | 768 | 930.06 | 1.95 | 0.4274 | 0.4588 | +ultrachat, no effect. Job 2110091 |

### DPO-Style Forget Loss, KL Retain Loss (Undertuned)

NPO loss on forget data routed through frozen remaining layers: -log sigmoid(-beta * (log_prob_current - log_prob_ref)). Compared against Run 24 (max-entropy KL forget, same config).

| Run | Layers | remove_coef | retain_coef | lr | dpo_beta | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 24 | 31→11 (step 4) | 5 | 200 | 2e-4 | - | 768 | 186.70 | 1.96 | 0.3629 | 0.3821 | max_entropy_kl (reference) |
| 45 | 31→11 (step 4) | 5 | 200 | 2e-4 | 0.1 | 768 | 5.40 | 0.69 | 0.4274 | 0.4523 | No effect |
| 46 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 4.95 | 0.69 | 0.4309 | 0.4525 | No effect |

### DPO-Style Forget Loss, NLL Retain Loss (Weak Unlearning)

NPO forget through frozen remaining layers + NLL retain (standard NPO retain). Compared against Run 29 (max-entropy KL forget + NLL retain).

Deeper layers saturate more slowly. For one early run:
  - Layer 31: saturates by step 8
  - Layer 15: saturates by step ~600 (12 gradient updates with signal)
  - Layer 11: never saturated, still at ~0.5 at step 760

| Run | Layers | remove_coef | retain_coef | lr | dpo_beta | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|----------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 29 | 31→11 (step 4) | 5 | 200 | 2e-4 | - | 768 | 1.70 | 1.87 | 0.3986 | 0.4519 | max_entropy_kl + nll_retain (reference) |
| 47 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.39 | 0.54 | 0.4113 | 0.4416 | nll_retain, Job 2113196, forget_loss saturates to 0 by ~step 8 each phase |
| 48 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.64 | 0.52 | 0.4159 | 0.4412 | nll_retain, dpo_fullmodel (per-seq NPO), Job 2114350 |
| 49 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.45 | 0.70 | 0.4136 | 0.4442 | nll_retain, per-token NPO, Job 2114673 |
| 50 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.45 | 0.68 | 0.4147 | 0.4412 | nll_retain, per-token NPO, forget_layer_scale=5, Job 2116051 |
| 51 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.46 | 0.68 | 0.4067 | 0.4397 | nll_retain, per-token NPO, forget_layer_scale=10, Job 2115652 |
| 52 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.58 | 0.67 | 0.4009 | 0.4368 | nll_retain, per-token NPO, forget_layer_scale=50, Job 2115645 |
| 53 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 2.06 | 0.67 | 0.3986 | 0.4326 | nll_retain, per-token NPO, forget_layer_scale=200, Job 2125918 |
| 54 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | 768 | 1.52 | 0.68 | 0.3986 | 0.4298 | nll_retain, per-token NPO, forget_layer_scale=500, Job 2125919 |
| 55 | 31→11 (step 4) | 5 | 200 | 2e-4 | 50.0 | 768 | 1.47 | 0.91 | 0.4101 | 0.4434 | nll_retain, per-token NPO, dpo_beta=50, forget_layer_scale=50, Job 2125920 |
| 56 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | ~1509 | 1.36 | 0.02 | 0.3998 | 0.4419 | nll_retain, per-token NPO, fls=50, deep_layer_step_scale=4, Job 2125951 |
| 57 | 31→11 (step 4) | 5 | 200 | 2e-4 | 10.0 | ~2500 | 0.85 | 0.004 | 0.3952 | 0.4422 | nll_retain, per-token NPO, fls=50, deep_layer_step_scale=8, Job 2125953 |

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

### Sequential SFT Tampering (eval every 10 steps, 512 examples, lr=2e-5, 1 epoch)

| Model | Step 0 | Step 10 | Step 20 | Step 30 | Step 40 | Step 50 | Step 60 | Step 70 | Step 80 | Step 90 | Step 100 | Step 110 |
|-------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| Baseline (target) | 0.4297 | - | - | - | - | - | - | - | - | - | - | - |
| Muon+KW (rm5/ret2, L31-11) | 0.2903 | 0.4090 | 0.4101 | 0.4263 | 0.4343 | 0.4332 | 0.4217 | 0.4182 | 0.4205 | 0.4251 | 0.4309 | 0.4320 |
| NLL Retain (rm2/ret200, L31-1s2) | 0.3975 | 0.4136 | 0.4332 | 0.4309 | 0.4332 | 0.4332 | 0.4424 | 0.4389 | 0.4389 | 0.4424 | 0.4435 | 0.4412 |
| SFT L2+UC (rm14/ret1, L31-11) | 0.2949 | 0.4182 | 0.4228 | 0.4597 | 0.4447 | 0.4401 | 0.4389 | 0.4320 | 0.4320 | 0.4263 | 0.4251 | 0.4274 |
| Muon+KW+L2SP (rm5/ret2, L31-11) | 0.3848 | 0.4113 | 0.3998 | 0.4401 | 0.4401 | 0.4435 | 0.4412 | 0.4424 | 0.4459 | 0.4435 | 0.4424 | 0.4435 |

All four models show rapid recovery by step 10-30. Plots saved in `runs/tamper_attack_muon_kw_rm5_ret2/`, `runs/tamper_attack_nllret_rm2_ret200/`, `runs/tamper_attack_sft_l2_uc_rm14_ret1/`, `runs/tamper_attack_muon_kw_l2sp/`.
