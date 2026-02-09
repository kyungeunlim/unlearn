# FSDP2 Migration Status

## Problem

Sequential unlearn SFT (`sequential_unlearn_sft.py`) relies on direct `param.grad` access for layer-selective gradient manipulation. FSDP2 (torchrun + DTensor) makes `param.grad` return `None` for all parameters because gradients are managed through DTensor infrastructure.

### Error observed (Job 2208795)

Training runs but produces no learning. All gradient norms are zero and no parameters are frozen:

```
step 0 | layer 31 (phase 1/6) | retain_loss: 0.0000 | forget_loss: 2.1715 | retain_grad_norm: 0.0000 | forget_grad_norm: 0.0000 | params with grad: 0 | frozen: 0
```

Result: WMDP Bio Robust 0.4320 (baseline), MMLU 0.4510 (baseline) -- no unlearning occurred.

### Earlier error (Job 2208022)

Before the DTensor fix, `_compute_grad_norm` crashed:

```
File "sequential_unlearn_sft.py", line 317, in _compute_grad_norm
    total_norm_sq += param.grad.detach().float().pow(2).sum()
File "torch/distributed/tensor/_dispatch.py", line 381
    assert isinstance(args[0], dtensor.DTensor)
AssertionError
```

Fixed by adding `.to_local()` before `.float()`, but this only addressed the crash -- the underlying `param.grad is None` issue means no gradients flow.

## Affected code paths in `sequential_unlearn_sft.py`

All of these check `param.grad is not None` and get `None` with FSDP2:

1. **`_freeze_and_log`** (line ~283): Zeros gradients for non-target layers. With FSDP2, finds 0 params with grad, freezes 0.
2. **`_compute_grad_norm`** (line ~312): Computes per-layer gradient norms. Returns 0.0.
3. **`training_step`** retain grad save (line ~381): `retain_grads[name] = param.grad.clone()` -- saves nothing.
4. **`training_step`** gradient combination (line ~424): Adding retain grads to forget grads -- no-op.
5. **`training_step`** same-sign filtering (line ~398): Also no-op.

## What needs to change for FSDP2

With FSDP2/DTensor, gradients are sharded DTensors. Options:

- Use `param.grad.to_local()` to access the local shard (but need to confirm `param.grad` is not None with FSDP2 -- it may require `torch.distributed.fsdp.fully_shard` specific configuration)
- Use `param.grad.full_tensor()` to gather the full gradient (expensive, defeats sharding purpose)
- Register backward hooks to intercept gradients before FSDP2 processes them
- Use FSDP2's `set_requires_gradient_sync` or similar APIs to control gradient reduction

## Algorithms still on FSDP1 (accelerate launch)

These scripts use `accelerate launch` with `fsdp_config_2node.yaml` and work correctly:

- `sequential_unlearn_sft.py` -- the main sequential unlearning trainer (needs FSDP1 for `param.grad` access)

## Scripts already migrated to FSDP2 (torchrun) -- broken for sequential unlearn

- `run_sequential_unlearn_sft_topk_muon.sbatch` (reverted below)
- `run_sequential_unlearn_sft_keyword.sbatch`
- `run_sequential_unlearn_sft_nokw.sbatch`

## Root cause

The `TrainingArguments` in `sequential_unlearn_sft.py` had `"fsdp_version": 2` in `fsdp_config`, which forces FSDP2 even when using `accelerate launch`. Removed this to fall back to FSDP1.

Additionally, `muon_lr` was removed from `SequentialSftUnlearnConfig` -- the `MuonAdamW` optimizer now uses a single `lr` with `adjust_lr_fn="match_rms_adamw"` for internal scaling.

## Current workaround

Removed `"fsdp_version": 2` from `TrainingArguments.fsdp_config` and `--muon_lr` from sbatch scripts.
