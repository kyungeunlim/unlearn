# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Planned Runs

| Run | remove_coef | orth_coef | Rationale |
|-----|-------------|-----------|-----------|
| 1   | 15          | 5         | Low intervention baseline |
| 2   | 23          | 10        | Default values |
| 3   | 30          | 15        | Moderately higher |
| 4   | 40          | 20        | High intervention (expect damage) |

## Results

| remove_coef | orth_coef | steps | WMDP | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|-------------|-----------|-------|------|------|-------------|---------|-----------|-------|
| 15          | 5         | 32    | -    | -    | -           | -       | -         | Not trained |
| 23          | 10        | 32    | -    | -    | -           | -       | -         | Model incomplete |
| 30          | 15        | 32    | 26.55% | 35.14% | 4.30      | 0.18    | 0.99      | |
| 30          | 15        | 512   | pending | pending | -       | -       | -         | Job 2018590 |
| 40          | 20        | 32    | 26.63% | 35.39% | -         | -       | -         | Log truncated |
