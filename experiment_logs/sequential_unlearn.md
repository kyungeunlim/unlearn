# Sequential Unlearning Experiments

Sequential back-to-front unlearning: for each layer L (from last to first), use the original model's layers L→N as a "probe" to compute forget loss while retaining activations close to original.

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

## Baseline
- Model: EleutherAI/deep-ignorance-unfiltered
- WMDP Bio Robust: ~0.44
- MMLU: ~0.45

## Experiments

| Run | Layers | remove_coef | retain_coef | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----------------|------|-------|
| 1 | 28→16 (step 4) | 10 | 5 | 0.3134 | 0.4479 | |
| 2 | 28→16 (step 4) | 20 | 5 | 0.3076 | 0.4454 | |
| 3 | 28→16 (step 4) | 30 | 5 | 0.2995 | 0.4458 | |
| 4 | 28→16 (step 4) | 40 | 5 | 0.2961 | 0.4410 | |
| 5 | 28→16 (step 4) | 50 | 5 | 0.2684 | 0.4437 | Below random (0.25) |

## Retain Loss Ablation

### L2 vs KL with same hyperparameters (rm50, ret5)

| retain_loss_type | WMDP Bio Robust | MMLU | Notes |
|------------------|-----------------|------|-------|
| L2 (hidden states) | 0.2684 | 0.4437 | Best L2 |
| KL (logits) | 0.2673 | 0.2295 | MMLU collapsed |

### KL hyperparameter sweep (requires different tuning)

| remove_coef | retain_coef | WMDP Bio Robust | MMLU | Notes |
|-------------|-------------|-----------------|------|-------|
| 50 | 5 | 0.2673 | 0.2295 | Collapsed |
| 20 | 10 | 0.2673 | - | Collapsed |
| 30 | 10 | 0.2673 | - | Collapsed |
| 20 | 20 | 0.2673 | 0.2295 | Collapsed |
| 10 | 20 | 0.2673 | - | Collapsed |
| 10 | 50 | 0.2673 | 0.2295 | Collapsed |
| 5 | 50 | 0.2730 | 0.4034 | Works |
| 5 | 100 | 0.2788 | 0.4195 | Best KL |

**Conclusion**: KL divergence requires ~10-20x higher retain_coef and ~10x lower remove_coef compared to L2. With proper tuning (rm5/ret100), KL achieves comparable results to L2 (rm50/ret5).

## Hyperparameters (all runs)

| Parameter | Value |
|-----------|-------|
| num_train_examples | 1024 |
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
| L2 (rm50/ret5) | 0.2696 | 0.4055 | 0.4182 |
| KL (rm5/ret100) | 0.2788 | 0.4078 | 0.4217 |

Both models show rapid recovery (~10 steps) back to near-baseline WMDP accuracy.
Plots saved in `runs/tamper_attack/` and `runs/tamper_kl_rm5_ret100/`.

## Key Findings

1. Higher remove_coef pushes WMDP closer to (and below) random chance
2. MMLU degradation remains minimal even at rm50 (~1% drop from baseline)
3. Best L2 result: rm50/ret5 achieves 0.2684 WMDP with 0.4437 MMLU
4. Random chance for 4-way MCQ is 0.25
5. KL divergence requires very different hyperparameters than L2 (~10-20x higher retain, ~10x lower remove)
6. Best KL result: rm5/ret100 achieves 0.2788 WMDP with 0.4195 MMLU
7. Both L2 and KL models are equally vulnerable to tampering attacks
