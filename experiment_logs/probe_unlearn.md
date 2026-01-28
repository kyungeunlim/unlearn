# Probe Unlearning Experiments

## Experiment Settings

| Job ID | num_train_examples | lora_r | layers | remove_coef | retain_coef | pdbs | lr | Steps | Status |
|--------|-------------------|--------|--------|-------------|-------------|------|------|-------|--------|
| 2042020 | 1024 | 16 | 8,12,16,20,24,28 | 5.0 | 5.0 | 4 | 1e-3 | 32 | Failed - missing args |
| 2042021 | 1024 | 16 | 8,12,16,20,24,28 | 5.0 | 5.0 | 4 | 1e-3 | 32 | Failed - no_grad blocked gradients |
| 2042022 | 1024 | 16 | 8,12,16,20,24,28 | 5.0 | 5.0 | 4 | 1e-3 | 32 | Running |

## Results

| Job ID | WMDP Bio Robust | MMLU | Retain Loss (final) | Forget Loss (final) | Notes |
|--------|-----------------|------|---------------------|---------------------|-------|

## Training Progress Notes
