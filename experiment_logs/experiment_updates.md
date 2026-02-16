# Experiment Updates

| Date | Description |
|------|-------------|
| 2026-02-16 | Hardcoded models to bfloat16 to fix some bug, in tamper attack with plot and in worker utils |
| 2026-02-15 | Fixed default `layers` in all algorithm configs from `[5, 10, 15, 20, 25, 30]` to `list(range(32))`. The old default meant layer 31 never received gradient signal and was never modified, despite `lora_all_layers=True` applying LoRA adapters to all 32 layers. All existing `_all_L32` models (r=2 through r=512) have layer 31 identical to the base model. The `no_L31` ablation was identical to the "full" model. Affected files: `orth_circuit_breakers.py`, `base_unlearn.py`, `checkpoint_transfer_unlearn.py`, `lens_unlearn.py`, `lens_unlearn_muon.py`, `lens_unlearn_sft.py`, `lora_checkpoint_transfer.py`, `max_update_unlearn.py`, `weight_partition_unlearn.py`, `sequential_unlearn.py`. |
