"""Compute mean effective rank of parameter update delta for linear modules.

Compares an unlearned model checkpoint against the base model to measure the
effective rank of weight deltas in all 2D linear layers.

Usage:
    python -m unlearn.scripts.compute_erank --model_path /path/to/unlearned/model
    python -m unlearn.scripts.compute_erank --model_path /path/to/model \
        --base_model EleutherAI/deep-ignorance-unfiltered
"""

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from simple_parsing import ArgumentParser
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from unlearn.utils.math import effective_rank


@dataclass
class ErankConfig:
    model_path: str = ""
    base_model: str = "EleutherAI/deep-ignorance-unfiltered"


def resolve_safetensor_paths(model_id: str) -> list[str]:
    """Resolve safetensor file paths for a model (local dir or HF cache)."""
    model_path = Path(model_id)

    # Local directory
    if model_path.is_dir():
        single = model_path / SAFE_WEIGHTS_NAME
        if single.exists():
            return [str(single)]
        index_file = model_path / SAFE_WEIGHTS_INDEX_NAME
        if index_file.exists():
            index = json.loads(index_file.read_text())
            shard_files = sorted(set(index["weight_map"].values()))
            return [str(model_path / f) for f in shard_files]
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    # HuggingFace cache
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == model_id:
            snapshot_dir = Path(repo.repo_path) / "snapshots"
            if snapshot_dir.exists():
                snapshots = sorted(snapshot_dir.iterdir())
                if snapshots:
                    return resolve_safetensor_paths(str(snapshots[-1]))
    raise FileNotFoundError(f"Model {model_id} not found locally or in HF cache")


def build_tensor_index(shard_paths: list[str]) -> dict[str, str]:
    """Map tensor keys to their shard file paths."""
    index = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                index[key] = path
    return index


def is_linear_weight(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    skip = ("embed", "layernorm", "layer_norm", "norm", "lm_head")
    return not any(s in name.lower() for s in skip)


def main():
    parser = ArgumentParser()
    parser.add_arguments(ErankConfig, dest="cfg")
    cfg = parser.parse_args().cfg

    print(f"Unlearned model: {cfg.model_path}")
    print(f"Base model: {cfg.base_model}")

    unlearned_paths = resolve_safetensor_paths(cfg.model_path)
    base_paths = resolve_safetensor_paths(cfg.base_model)

    print("Building tensor indices...")
    unlearned_index = build_tensor_index(unlearned_paths)
    base_index = build_tensor_index(base_paths)

    # Find all 2D linear weight keys present in both models
    all_keys = sorted(set(unlearned_index.keys()) & set(base_index.keys()))
    linear_keys = [k for k in all_keys if is_linear_weight(k)]

    # Filter to 2D
    valid_keys = []
    for k in linear_keys:
        with safe_open(unlearned_index[k], framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
            if t.ndim == 2:
                valid_keys.append(k)

    print(f"Found {len(valid_keys)} linear weight matrices\n")

    eranks = []
    for i, key in enumerate(valid_keys):
        with safe_open(unlearned_index[key], framework="pt", device="cpu") as f:
            w_unlearned = f.get_tensor(key).float()
        with safe_open(base_index[key], framework="pt", device="cpu") as f:
            w_base = f.get_tensor(key).float()

        delta = w_unlearned - w_base

        if delta.abs().max().item() < 1e-12:
            print(f"  [{i+1}/{len(valid_keys)}] {key}: no change", flush=True)
            continue

        er = effective_rank(delta)
        eranks.append(er)
        print(
            f"  [{i+1}/{len(valid_keys)}] {key}: "
            f"shape={tuple(delta.shape)}, erank={er:.2f}",
            flush=True,
        )

        del w_unlearned, w_base, delta

    print(f"\n{'='*60}")
    print(f"Linear modules with non-zero delta: {len(eranks)} / {len(valid_keys)}")
    if eranks:
        print(f"Mean effective rank: {sum(eranks)/len(eranks):.2f}")
        print(f"Min effective rank:  {min(eranks):.2f}")
        print(f"Max effective rank:  {max(eranks):.2f}")
    else:
        print("No weight changes detected.")


if __name__ == "__main__":
    main()
