# Checkpoint Transfer Hyperparameter Tuning

**Goal:** Find the boundary between unlearning both evals and unlearning neither.

**Fixed settings:**
- `num_train_examples`: 1024
- `pdbs`: 4 (per device batch size)
- `affine`: True (loaded from EleutherAI/affine-checkpoint-transfer)
- `KL divergence retain`: True
- `lr`: 1e-3
- `layers`: [5, 10, 15, 20, 25, 30]

## Baselines

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |
| Checkpoint (source activations) | 24.19% | 27.78% |

## Checkpoint Sweep

Finding the latest checkpoint with near-random WMDP Bio performance:

| Checkpoint | WMDP Bio | Notes |
|------------|----------|-------|
| global_step38144 | 24.19% | Current source |
| global_step50064 | 25.23% | ~Random |
| global_step54832 | 26.50% | ~Random (latest viable) |
| global_step57216 | 29.03% | Learning starts |
| global_step60792 | 30.41% | |
| global_step70328 | 33.76% | |
| global_step81056 | 34.10% | |
| global_step91784 | 38.02% | |

**Recommendation:** global_step54832 is ~17k steps further into training but still has effectively random WMDP performance.

## Notes

- Higher remove_coef = stronger push towards checkpoint activations (more unlearning)
- Higher retain_coef = stronger KL divergence preservation (less capability loss)
- Looking for the sweet spot where WMDP drops but MMLU stays high

## Runs

| Job ID | retain_coef | remove_coef | Steps | retain_kl_loss | cb_loss | WMDP Bio | MMLU STEM |
|--------|-------------|-------------|-------|----------------|---------|----------|-----------|
| 2021158 | 2 | 5 | 256 | 0.0190 | 0.8912 | 0.3952 | 0.3619 |
| 2021163 | 2 | 20 | 256 | 0.0481 | 0.8683 | 0.3721 | 0.3524 |
| 2021160 | 5 | 10 | 256 | 0.0153 | 0.8692 | 0.3744 | 0.3552 |
| 2024175 | 5 | 2 | 256 | 0.0053 | 0.8858 | 0.3952 | 0.3647 |
| 2024177 | 5 | 1 | 256 | 0.0031 | 0.8890 | 0.4067 | 0.3619 |
| 2024176 | 10 | 5 | 256 | 0.0063 | 0.8807 | 0.3952 | 0.3635 |
| 2024178 | 20 | 10 | 256 | 0.0056 | 0.8725 | 0.3906 | 0.3612 |
| 2021162 | 10 | 20 | 256 | 0.0171 | 0.8710 | 0.3744 | 0.3612 |
| 2025807 | 2 | 30 | 256 | 0.0593 | 0.8750 | 0.3687 | 0.3457 |
| 2025808 | 2 | 40 | 256 | 0.0641 | 0.8875 | 0.3629 | 0.3432 |
| 2025809 | 2 | 50 | 256 | 0.0737 | 0.8725 | 0.3687 | 0.3454 |
| 2025810 | 2 | 60 | 256 | 0.0697 | 0.8791 | 0.3606 | 0.3416 |
| 2026007 | 2 | 100 | 256 | 0.0660 | 0.8843 | 0.3606 | 0.3416 |
| 2026828 | 2 | 100 | 512 | 0.0608 | 0.7283 | 0.3260 | 0.3365 |
| 2028287 | 2 | 100 | 2048 | | | | |
| 2028225 | 2 | 50 | 2048 | | | | |

### Batch 1 Observations
- All configurations produce similar WMDP (0.37-0.40) and MMLU (0.35-0.36)
- Higher remove_coef with low retain_coef (2,20) slightly worse MMLU
- Need baseline model performance for comparison
- Need to explore more extreme configs to find boundary

### Batch 2 Observations
- Even with lowest remove_coef (1), WMDP only rises to 0.4067 (vs 0.39-0.40 for others)
- All configs converge to similar WMDP (0.39-0.41) and MMLU (0.36-0.37)
- Unlearning effect seems to plateau early; diminishing returns from higher remove_coef
- Need baseline model performance to understand "no unlearning" case

### Batch 3 Observations (retain=2, varying remove)
- MMLU drops steadily as remove_coef increases: 0.36 → 0.34
- WMDP also drops: 0.40 → 0.36
- remove=60 and remove=100 give identical results (plateau)
- retain_kl_loss increases with higher remove_coef (0.06-0.07 vs 0.02-0.05 earlier)

### 2x Steps (retain=2, remove=100, 512 steps)
- WMDP drops further: 0.3606 → 0.3260 (closer to checkpoint baseline of 0.2419)
- MMLU drops slightly: 0.3416 → 0.3365 (still above checkpoint baseline of 0.2778)
- cb_loss drops significantly: 0.8843 → 0.7283 (model moving closer to checkpoint)

