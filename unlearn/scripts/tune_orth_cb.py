#!/usr/bin/env python3
"""Hyperparameter tuning script for orthogonal circuit breakers.

Tunes orth_coef and remove_coef and evaluates on WMDP/MMLU.
Results are logged to a markdown file.
"""

import argparse
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class TuningConfig:
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    num_train_examples: int = 1024
    pdbs: int = 4
    retain_coef: float = 5.0
    lm_eval_include_path: str = str(PROJECT_ROOT / "unlearn" / "lm_eval_tasks")
    results_file: str = str(PROJECT_ROOT / "runs" / "tuning_results.md")
    models_dir: str = str(PROJECT_ROOT / "models")


def run_training(
    orth_coef: float, remove_coef: float, save_name: str, cfg: TuningConfig
) -> bool:
    """Run training with specified coefficients."""
    model_path = os.path.join(
        cfg.models_dir,
        cfg.model_name.replace("/", "_") + "_" + save_name,
    )
    cmd = [
        "python",
        "-m",
        "unlearn.algorithm.orth_circuit_breakers",
        f"--orth_coef={orth_coef}",
        f"--remove_coef={remove_coef}",
        f"--retain_coef={cfg.retain_coef}",
        f"--num_train_examples={cfg.num_train_examples}",
        f"--pdbs={cfg.pdbs}",
        f"--model_name={cfg.model_name}",
        f"--save_path={model_path}",
    ]

    print(f"Running training command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        env=env,
    )
    return result.returncode == 0


def run_wmdp_eval(model_path: str, cfg: TuningConfig) -> dict | None:
    """Run WMDP evaluation."""
    cmd = [
        "python",
        "-m",
        "unlearn.evaluation.eval_wmdp_robust",
        f"--model_path={model_path}",
        f"--include_path={cfg.lm_eval_include_path}",
        "--batch_size=8",
    ]

    print(f"Running WMDP eval command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"WMDP eval failed: {result.stderr}")
        return None

    for line in result.stdout.split("\n"):
        if "acc,none" in line or "acc_norm" in line:
            try:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    return {"wmdp_acc": float(parts[1].strip())}
            except ValueError:
                continue

    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if "wmdp_bio_robust" in line.lower():
            for j in range(i, min(i + 10, len(lines))):
                if "acc" in lines[j].lower():
                    parts = lines[j].split(":")
                    if len(parts) == 2:
                        try:
                            return {"wmdp_acc": float(parts[1].strip())}
                        except ValueError:
                            continue

    print(f"Could not parse WMDP results from: {result.stdout}")
    return {"wmdp_acc": -1.0}


def run_mmlu_eval(model_path: str, cfg: TuningConfig) -> dict | None:
    """Run MMLU evaluation."""
    cmd = [
        "python",
        "-m",
        "unlearn.evaluation.eval_mmlu",
        f"--model_path={model_path}",
        "--batch_size=8",
    ]

    print(f"Running MMLU eval command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"MMLU eval failed: {result.stderr}")
        return None

    for line in result.stdout.split("\n"):
        if "acc" in line.lower():
            parts = line.strip().split(":")
            if len(parts) == 2:
                try:
                    return {"mmlu_acc": float(parts[1].strip())}
                except ValueError:
                    continue

    print(f"Could not parse MMLU results from: {result.stdout}")
    return {"mmlu_acc": -1.0}


def write_results_header(results_file: str):
    """Initialize the results markdown file."""
    with open(results_file, "w") as f:
        f.write("# Orthogonal Circuit Breaker Tuning Results\n\n")
        f.write(f"Started: {datetime.now().isoformat()}\n\n")
        f.write(
            "| orth_coef | remove_coef | WMDP (lower=better) | "
            "MMLU (higher=better) | Notes |\n"
        )
        f.write(
            "|-----------|-------------|---------------------"
            "|----------------------|-------|\n"
        )


def append_result(
    results_file: str,
    orth_coef: float,
    remove_coef: float,
    wmdp_acc: float,
    mmlu_acc: float,
    notes: str = "",
):
    """Append a result row to the markdown file."""
    with open(results_file, "a") as f:
        f.write(
            f"| {orth_coef:.2f} | {remove_coef:.2f} | {wmdp_acc:.4f} "
            f"| {mmlu_acc:.4f} | {notes} |\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--num_train_examples", type=int, default=1024)
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument("--retain_coef", type=float, default=5.0)
    args = parser.parse_args()

    cfg = TuningConfig(
        model_name=args.model_name,
        num_train_examples=args.num_train_examples,
        pdbs=args.pdbs,
        retain_coef=args.retain_coef,
    )

    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.results_file), exist_ok=True)

    orth_coefs = [0.0, 0.5, 1.0, 2.0, 5.0]
    remove_coefs = [3.0, 5.0, 7.0, 10.0]

    write_results_header(cfg.results_file)

    best_score = float("-inf")
    best_params = None

    for remove_coef in remove_coefs:
        for orth_coef in orth_coefs:
            save_name = f"orth{orth_coef}_rm{remove_coef}"
            model_path = os.path.join(
                cfg.models_dir,
                cfg.model_name.replace("/", "_") + "_" + save_name,
            )

            print(f"\n{'='*60}")
            print(f"Training: orth_coef={orth_coef}, remove_coef={remove_coef}")
            print(f"{'='*60}\n")

            success = run_training(orth_coef, remove_coef, save_name, cfg)
            if not success:
                append_result(
                    cfg.results_file,
                    orth_coef,
                    remove_coef,
                    -1.0,
                    -1.0,
                    "Training failed",
                )
                continue

            print(f"\nEvaluating model at {model_path}")

            wmdp_results = run_wmdp_eval(model_path, cfg)
            mmlu_results = run_mmlu_eval(model_path, cfg)

            wmdp_acc = wmdp_results.get("wmdp_acc", -1.0) if wmdp_results else -1.0
            mmlu_acc = mmlu_results.get("mmlu_acc", -1.0) if mmlu_results else -1.0

            score = mmlu_acc - wmdp_acc
            notes = ""
            if score > best_score and wmdp_acc >= 0 and mmlu_acc >= 0:
                best_score = score
                best_params = (orth_coef, remove_coef)
                notes = "**NEW BEST**"

            append_result(
                cfg.results_file, orth_coef, remove_coef, wmdp_acc, mmlu_acc, notes
            )

    with open(cfg.results_file, "a") as f:
        f.write("\n\n## Best Parameters\n\n")
        if best_params:
            f.write(f"- orth_coef: {best_params[0]}\n")
            f.write(f"- remove_coef: {best_params[1]}\n")
            f.write(f"- Score (MMLU - WMDP): {best_score:.4f}\n")
        else:
            f.write("No successful runs completed.\n")

    print(f"\n\nResults written to: {cfg.results_file}")
    if best_params:
        print(
            f"Best parameters: orth_coef={best_params[0]}, remove_coef={best_params[1]}"
        )


if __name__ == "__main__":
    main()
