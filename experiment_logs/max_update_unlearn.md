# Max Update Unlearning

Model: EleutherAI/deep-ignorance-unfiltered, SFT

## Results

| Run | retain_coef | update_coef | lr | num_examples | epochs | steps | sft_loss (final) | update_norm (final) | WMDP Bio Robust | MMLU |
|-----|-------------|-------------|------|--------------|--------|-------|------------------|---------------------|-----------------|------|
| Baseline | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 |
| Job 2134097 | 1.0 | 0.0 | 2e-4 | 512 | 1 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134098 | 1.0 | 10.0 | 2e-4 | 512 | 1 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134099 | 1.0 | 50.0 | 2e-4 | 512 | 1 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134100 | 1.0 | 100.0 | 2e-4 | 512 | 1 | 16 | ~2.23 | ~0.0 | 0.4297 | 0.4510 |

Jobs 2134097-2134100: update_norm was 0 due to /N normalization bug (dividing by ~3.9B params). Fixed.

| Run | retain_coef | update_coef | lr | num_examples | epochs | steps | sft_loss (final) | update_norm (final) | WMDP Bio Robust | MMLU |
|-----|-------------|-------------|------|--------------|--------|-------|------------------|---------------------|-----------------|------|
| Job 2270984 | 1.0 | 0.0 | 2e-4 | 1024 | 10 | 320 | | | | |
| Job 2270985 | 1.0 | 0.01 | 2e-4 | 1024 | 10 | 320 | | | | |
| Job 2270986 | 1.0 | 0.1 | 2e-4 | 1024 | 10 | 320 | | | | |
| Job 2270987 | 1.0 | 1.0 | 2e-4 | 1024 | 10 | 320 | | | | |
| Job 2270988 | 1.0 | 10.0 | 2e-4 | 1024 | 10 | 320 | | | | |
