# Lens Unlearn SFT Experiments

## Experiment Settings

| Job ID | num_train_examples | remove_coef | retain_coef (actual) | pdbs | lr | Model | Steps | Status |
|--------|-------------------|-------------|---------------------|------|------|-------|-------|--------|
| 2042014 | 1024 | 5.0 | 0.0 (SFT has no ref model) | 2 | 1e-4 | EleutherAI/deep-ignorance-unfiltered | 32 | Completed |

## Results

| Job ID | WMDP Bio Robust | MMLU | Training Loss | Forget Loss (final) | Notes |
|--------|-----------------|------|---------------|---------------------|-------|
| 2042014 | FAILED | FAILED | 4.83 | 1.02 | Evals failed: `TypeError: GPTNeoXForCausalLM.__init__() got an unexpected keyword argument 'dtype'` - need to fix eval script |

## Training Progress Notes (Job 2042014)

- Initial forget_loss: 2.16
- Final forget_loss: 1.02
- retain_coef was auto-set to 0 because "Retain loss requires a frozen reference model"
- Model saved to: `models/EleutherAI/deep-ignorance-unfiltered_lens_ex1024_sft_sft`
- WandB run: https://wandb.ai/eleutherai/huggingface/runs/8kofqs5s
