# Tuned Lens Unlearning - 8192 Examples

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens (HuggingFace)
- **Training examples**: 8192 (256 steps)
- **GPUs**: 4x GH200 (distributed training via torchrun)
- **Batch size**: 4 per device, grad accumulation 2 (effective batch 32)
- **Learning rate**: 0.001
- **LoRA rank**: 16
- **Target layers**: [5, 10, 15, 20, 25, 30]
- **Coefficients**: retain_coef=5.0, remove_coef=5.0

## Training Metrics (Final)
- **retain_loss**: 1.19
- **forget_loss**: 1.04
- **Training time**: ~22 minutes

## Evaluation Results

| Benchmark | Score | Stderr |
|-----------|-------|--------|
| WMDP Bio Robust | 23.16% | ±1.44% |
| MMLU STEM | 34.35% | ±0.83% |

### WMDP Bio Robust Breakdown
| Subtask | Score |
|---------|-------|
| bioweapons_and_bioterrorism | 25.26% |
| dual_use_virology | 25.00% |
| enhanced_potential_pandemic_pathogens | 18.63% |
| expanding_access_to_threat_vectors | 23.81% |
| reverse_genetics_and_easy_editing | 23.12% |
| viral_vector_research | 23.17% |

## Notes
- WMDP Bio score near random chance (25%) indicates successful unlearning
- MMLU STEM retained at reasonable level (34.35%)
- Model saved to: `models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0`
