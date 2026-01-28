# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Baselines

| Model | WMDP Bio | MMLU STEM | Full MMLU |
|-------|----------|-----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% | ~43% |

## Metric Interpretation

**orth_loss**: Average pairwise cosine similarity between forget item representations in each batch. Measures whether forget items collapse to the same direction or remain diverse.
- **~1.0** = bad (all forget items aligned to same direction, vulnerable to single-direction attacks)
- **~0.5** = moderate diversity
- **~0** = good (orthogonal forget directions)

## Results

| remove | orth | retain | steps | WMDP | MMLU STEM | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|------|-----------|------|-------------|---------|-----------|-------|
| 15     | 5    | 2      | 32    | -    | -         | -    | -           | -       | -         | Not trained |
| 23     | 10   | 2      | 32    | -    | -         | -    | -           | -       | -         | Model incomplete |
| 30     | 15   | 2      | 32    | 26.55% | 35.14%  | -    | 4.30        | 0.18    | 0.99      | |
| 30     | 15   | 2      | 512   | 24.12% | 22.14%  | 23.76% | 1.08        | 0.015   | 0.05      | Severe capability damage |
| 40     | 20   | 2      | 32    | 26.63% | 35.39%  | -    | -           | -       | -         | Log truncated |
| 15     | 15   | 2      | 512   | 23.88% | 25.98%  | 29.90% | 1.01        | 0.03    | 0.04      | Capability damage |
| 30     | 100  | 2      | 32    | 39.2%  | 36.25%  | -    | 3.52        | 0.35    | 0.33      | Preserved capability, worse unlearning |
| 30     | 100  | 2      | 256   | 29.85% | 34.92%  | 42.18% | 2.95        | 0.10    | 0.09      | Good balance |
| 30     | 15   | 2      | 64    | 27.73% | 36.98%  | -    | 3.46        | 0.06    | 0.52      | Good balance, near-baseline MMLU |
| 30     | 15   | 2      | 128   | 24.35% | 30.03%  | -    | 2.15        | 0.02    | 0.25      | Better orth, some capability loss |
