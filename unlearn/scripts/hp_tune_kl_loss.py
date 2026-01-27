#!/usr/bin/env python3
"""Hyperparameter tuning script for KL divergence retain loss."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class HPConfig:
    retain_coef: float
    remove_coef: float
    num_train_examples: int = 1024
    pdbs: int = 2
    lr: float = 0.001


CONFIGS = [
    HPConfig(retain_coef=1, remove_coef=3),
    HPConfig(retain_coef=3, remove_coef=5),
    HPConfig(retain_coef=5, remove_coef=5),
    HPConfig(retain_coef=10, remove_coef=5),
    HPConfig(retain_coef=5, remove_coef=10),
    HPConfig(retain_coef=10, remove_coef=10),
]


def run_training(hp_cfg: HPConfig, output_dir: str, job_name: str) -> Optional[str]:
    """Run training with given hyperparameters."""
    save_name = f"kl_ret{hp_cfg.retain_coef}_rm{hp_cfg.remove_coef}"

    cmd = [
        sys.executable, "-m", "unlearn.algorithm.checkpoint_transfer_unlearn",
        "--num_train_examples", str(hp_cfg.num_train_examples),
        "--pdbs", str(hp_cfg.pdbs),
        "--lr", str(hp_cfg.lr),
        "--retain_coef", str(hp_cfg.retain_coef),
        "--remove_coef", str(hp_cfg.remove_coef),
        "--retain_kl_loss",
        "--save_name", save_name,
    ]

    print(f"\n{'='*60}")
    print(f"Training: retain_coef={hp_cfg.retain_coef}, remove_coef={hp_cfg.remove_coef}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None

    model_path = f"./models/EleutherAI/deep-ignorance-unfiltered_{save_name}"
    return model_path


def run_eval_wmdp(model_path: str, include_path: str) -> Optional[float]:
    """Run WMDP Bio Robust evaluation."""
    cmd = [
        sys.executable, "-m", "unlearn.evaluation.eval_wmdp_robust",
        "--model_path", model_path,
        "--include_path", include_path,
        "--batch_size", "4",
    ]

    print(f"\nRunning WMDP evaluation...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"WMDP eval failed: {result.stderr}")
        return None

    # Parse output to extract score
    for line in result.stdout.split("\n"):
        if "acc,none" in line or "acc_norm,none" in line:
            try:
                score = float(line.split(":")[-1].strip())
                return score * 100  # Convert to percentage
            except ValueError:
                continue

    print(f"Could not parse WMDP score from output:\n{result.stdout}")
    return None


def run_eval_mmlu(model_path: str) -> Optional[float]:
    """Run MMLU STEM evaluation."""
    cmd = [
        sys.executable, "-m", "unlearn.evaluation.eval_mmlu_stem",
        "--model_path", model_path,
        "--batch_size", "4",
    ]

    print(f"\nRunning MMLU evaluation...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"MMLU eval failed: {result.stderr}")
        return None

    # Parse output to extract score
    for line in result.stdout.split("\n"):
        if "acc,none" in line or "acc_norm,none" in line:
            try:
                score = float(line.split(":")[-1].strip())
                return score * 100
            except ValueError:
                continue

    print(f"Could not parse MMLU score from output:\n{result.stdout}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_idx",
        type=int,
        default=None,
        help="Run only specific config index (0-based)",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default="/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks",
        help="Path to lm_eval custom tasks",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation on existing models",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation (training only)",
    )
    run_args = parser.parse_args()

    output_dir = "./runs"
    os.makedirs(output_dir, exist_ok=True)

    configs_to_run = [CONFIGS[run_args.config_idx]] if run_args.config_idx is not None else CONFIGS

    results = []

    for i, hp_cfg in enumerate(configs_to_run):
        config_name = f"ret{hp_cfg.retain_coef}_rm{hp_cfg.remove_coef}"
        print(f"\n{'#'*60}")
        print(f"Config {i+1}/{len(configs_to_run)}: {config_name}")
        print(f"{'#'*60}")

        save_name = f"kl_ret{hp_cfg.retain_coef}_rm{hp_cfg.remove_coef}"
        model_path = f"./models/EleutherAI/deep-ignorance-unfiltered_{save_name}"

        if not run_args.skip_training:
            model_path = run_training(hp_cfg, output_dir, config_name)
            if model_path is None:
                results.append({
                    "retain_coef": hp_cfg.retain_coef,
                    "remove_coef": hp_cfg.remove_coef,
                    "wmdp": None,
                    "mmlu": None,
                    "status": "training_failed",
                })
                continue

        wmdp_score = None
        mmlu_score = None

        if not run_args.skip_eval and os.path.exists(model_path):
            wmdp_score = run_eval_wmdp(model_path, run_args.include_path)
            mmlu_score = run_eval_mmlu(model_path)

        results.append({
            "retain_coef": hp_cfg.retain_coef,
            "remove_coef": hp_cfg.remove_coef,
            "wmdp": wmdp_score,
            "mmlu": mmlu_score,
            "status": "completed",
        })

        print(f"\n{'='*60}")
        print(f"Results for retain_coef={hp_cfg.retain_coef}, remove_coef={hp_cfg.remove_coef}:")
        print(f"  WMDP Bio: {wmdp_score:.2f}%" if wmdp_score else "  WMDP Bio: N/A")
        print(f"  MMLU STEM: {mmlu_score:.2f}%" if mmlu_score else "  MMLU STEM: N/A")
        print(f"{'='*60}")

    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'retain_coef':<12} {'remove_coef':<12} {'WMDP Bio':<15} {'MMLU STEM':<15} {'Status':<15}")
    print("-" * 80)
    for r in results:
        wmdp_str = f"{r['wmdp']:.2f}%" if r['wmdp'] else "N/A"
        mmlu_str = f"{r['mmlu']:.2f}%" if r['mmlu'] else "N/A"
        print(f"{r['retain_coef']:<12} {r['remove_coef']:<12} {wmdp_str:<15} {mmlu_str:<15} {r['status']:<15}")


if __name__ == "__main__":
    main()
