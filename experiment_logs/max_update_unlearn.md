# Max Update Unlearning

Model: EleutherAI/deep-ignorance-unfiltered, SFT

## Results

| Run | retain_coef | update_coef | lr | num_examples | steps | sft_loss (final) | update_norm (final) | WMDP Bio Robust | MMLU |
|-----|-------------|-------------|------|--------------|-------|------------------|---------------------|-----------------|------|
| Baseline | - | - | - | - | - | - | - | 0.4297 | 0.4510 |
| Job 2134097 | 1.0 | 0.0 | 2e-4 | 512 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134098 | 1.0 | 10.0 | 2e-4 | 512 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134099 | 1.0 | 50.0 | 2e-4 | 512 | 16 | ~2.22 | ~0.0 | 0.4297 | 0.4510 |
| Job 2134100 | 1.0 | 100.0 | 2e-4 | 512 | 16 | ~2.23 | ~0.0 | 0.4297 | 0.4510 |
