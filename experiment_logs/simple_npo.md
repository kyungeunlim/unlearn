# SimpleNPO (SimNPO) Unlearning

Loss: `forget_loss = -logsigmoid(beta * CE_forget) * 2/beta; total = forget_loss + gamma * retain_loss`

Model: EleutherAI/deep-ignorance-unfiltered, LoRA r=16

## Results

| beta | gamma | lr | num_examples | steps | forget_ce (final) | retain_loss (final) | WMDP Bio Robust | MMLU |
|------|-------|----|-------------|-------|--------------------|---------------------|-----------------|------|
| - | - | - | - | - | - | - | 42.97% | 45.10% |
| 0.1 | 1.0 | 1e-3 | 1024 | 32 | 105.4 | 1.88 | 41.47% | 44.47% |
| 0.5 | 1.0 | 1e-3 | 1024 | 32 | 39.7 | 1.87 | 42.28% | 44.50% |
| 1.0 | 1.0 | 1e-3 | 1024 | 32 | 22.7 | 1.86 | 44.82% | 44.96% |
| 0.1 | 1.0 | 5e-3 | 1024 | 32 | 118.5 | 1.86 | 37.21% | 44.04% |
| 0.1 | 1.0 | 1e-3 | 4096 | 128 | | | | |
| 0.1 | 0.0 | 1e-3 | 4096 | 128 | | | | |
| 0.1 | 1.0 | 1e-2 | 1024 | 32 | | | | |
| 0.1 | 1.0 | 5e-3 | 4096 | 128 | | | | |
| 0.1 | 0.5 | 5e-3 | 1024 | 32 | | | | |
