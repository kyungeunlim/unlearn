"""Ablate LoRA update directions from a merged model.

Loads a base model and a merged (base + LoRA) model, computes the rank-r delta
for each LoRA-targeted weight matrix, then projects out the column space of that
delta from the merged weights. This zeros out the directions the LoRA learned to
use, which is more aggressive than simply reverting to the base model (it also
removes the base model's components in those directions).

The ablated model is saved to disk for downstream evaluation or tamper attacks.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEOX_ALL_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]


@dataclass
class AblateConfig:
    base_model: str = "EleutherAI/deep-ignorance-unfiltered"
    merged_model: str = ""
    output_path: str = ""
    lora_rank: int = 8
    target_modules: list[str] | None = None
    num_layers: int = 32


def get_weight_pairs(base_model, merged_model, ablate_cfg: AblateConfig):
    """Yield (param_name, base_param, merged_param) for each LoRA-targeted weight."""
    target_modules = ablate_cfg.target_modules or NEOX_ALL_MODULES
    for layer_idx in range(ablate_cfg.num_layers):
        for module_name in target_modules:
            if module_name in ("query_key_value", "dense"):
                param_name = (
                    f"gpt_neox.layers.{layer_idx}.attention.{module_name}.weight"
                )
            else:
                param_name = f"gpt_neox.layers.{layer_idx}.mlp.{module_name}.weight"

            base_param = base_model.state_dict()[param_name]
            merged_param = merged_model.state_dict()[param_name]
            yield param_name, base_param, merged_param


def ablate_directions(
    base_param: torch.Tensor, merged_param: torch.Tensor, rank: int
) -> torch.Tensor:
    """Project out the top-r left singular vectors of (merged - base) from merged.

    Given delta = merged - base with randomized SVD: delta ~ U @ S @ V^T,
    take the top-r columns of U and compute:
        ablated = (I - U_r @ U_r^T) @ merged
    This zeros out the LoRA output directions in the merged weight matrix.
    Uses torch.svd_lowrank for speed (full SVD is too slow on CPU for large matrices).
    """
    delta = (merged_param - base_param).float()
    U, S, V = torch.svd_lowrank(delta, q=3 * rank)
    U_r = U[:, :rank]

    projection = U_r @ (U_r.T @ merged_param.float())
    ablated = merged_param.float() - projection
    return ablated.to(merged_param.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--merged_model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--num_layers", type=int, default=32)
    args = parser.parse_args()

    ablate_cfg = AblateConfig(
        base_model=args.base_model,
        merged_model=args.merged_model,
        output_path=args.output_path,
        lora_rank=args.lora_rank,
        num_layers=args.num_layers,
    )

    print(f"Loading base model: {ablate_cfg.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        ablate_cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading merged model: {ablate_cfg.merged_model}", flush=True)
    merged_model = AutoModelForCausalLM.from_pretrained(
        ablate_cfg.merged_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print("Both models loaded.", flush=True)

    total_ablated = 0
    n_total = ablate_cfg.num_layers * len(ablate_cfg.target_modules or NEOX_ALL_MODULES)
    for param_name, base_param, merged_param in get_weight_pairs(
        base_model, merged_model, ablate_cfg
    ):
        ablated_param = ablate_directions(
            base_param, merged_param, ablate_cfg.lora_rank
        )
        merged_model.state_dict()[param_name].copy_(ablated_param)
        total_ablated += 1
        if total_ablated % 16 == 0:
            print(f"  Ablated {total_ablated}/{n_total} weight matrices", flush=True)

    print(f"\nAblated {total_ablated} weight matrices (rank {ablate_cfg.lora_rank})")

    output_path = Path(ablate_cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving ablated model to {output_path}")
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(ablate_cfg.merged_model)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
