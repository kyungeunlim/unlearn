#!/usr/bin/env python3
"""Run a single hyperparameter configuration: train and evaluate."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def append_to_results(
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
            f"| {orth_coef:.2f} | {remove_coef:.2f} | {wmdp_acc:.4f}"
            f" | {mmlu_acc:.4f} | {notes} |\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orth_coef", type=float, required=True)
    parser.add_argument("--remove_coef", type=float, required=True)
    parser.add_argument("--retain_coef", type=float, default=2.0)
    parser.add_argument("--num_train_examples", type=int, default=1024)
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    args = parser.parse_args()

    results_file = PROJECT_ROOT / "runs" / "tuning_results.md"
    models_dir = PROJECT_ROOT / "models"
    lm_eval_include_path = PROJECT_ROOT / "unlearn" / "lm_eval_tasks"

    os.makedirs(models_dir, exist_ok=True)

    save_name = f"orth{args.orth_coef}_rm{args.remove_coef}"
    model_save_path = models_dir / f"{args.model_name.replace('/', '_')}_{save_name}"

    print("=" * 60)
    print("Hyperparameter Configuration")
    print("=" * 60)
    print(f"orth_coef: {args.orth_coef}")
    print(f"remove_coef: {args.remove_coef}")
    print(f"retain_coef: {args.retain_coef}")
    print(f"num_train_examples: {args.num_train_examples}")
    print(f"pdbs: {args.pdbs}")
    print(f"model_name: {args.model_name}")
    print(f"save_name: {save_name}")
    print(f"model_save_path: {model_save_path}")
    print("=" * 60)
    sys.stdout.flush()

    # Step 1: Train
    print(f"\n[{datetime.now()}] Starting training...")
    train_cmd = [
        "python",
        "-m",
        "unlearn.algorithm.orth_circuit_breakers",
        f"--orth_coef={args.orth_coef}",
        f"--remove_coef={args.remove_coef}",
        f"--retain_coef={args.retain_coef}",
        f"--num_train_examples={args.num_train_examples}",
        f"--pdbs={args.pdbs}",
        f"--model_name={args.model_name}",
        f"--save_name={save_name}",
    ]
    print(f"Command: {' '.join(train_cmd)}")
    sys.stdout.flush()

    train_result = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    if train_result.returncode != 0:
        print(f"Training failed with return code {train_result.returncode}")
        append_to_results(
            str(results_file),
            args.orth_coef,
            args.remove_coef,
            -1.0,
            -1.0,
            "Training failed",
        )
        sys.exit(1)

    print(f"\n[{datetime.now()}] Training complete. Model saved to {model_save_path}")
    sys.stdout.flush()

    # Step 2: Evaluate WMDP
    print(f"\n[{datetime.now()}] Starting WMDP evaluation...")
    wmdp_cmd = [
        "python",
        "-m",
        "unlearn.evaluation.eval_wmdp_robust",
        f"--model_path={model_save_path}",
        f"--include_path={lm_eval_include_path}",
        "--batch_size=8",
    ]
    print(f"Command: {' '.join(wmdp_cmd)}")
    sys.stdout.flush()

    wmdp_result = subprocess.run(
        wmdp_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    print(wmdp_result.stdout)
    if wmdp_result.stderr:
        print(f"STDERR: {wmdp_result.stderr}")

    wmdp_acc = -1.0
    for line in wmdp_result.stdout.split("\n"):
        if "acc" in line.lower() and ":" in line:
            try:
                parts = line.split(":")
                val = float(parts[-1].strip())
                if 0 <= val <= 1:
                    wmdp_acc = val
                    break
            except ValueError:
                continue

    print(f"\n[{datetime.now()}] WMDP accuracy: {wmdp_acc}")
    sys.stdout.flush()

    # Step 3: Evaluate MMLU
    print(f"\n[{datetime.now()}] Starting MMLU evaluation...")
    mmlu_cmd = [
        "python",
        "-m",
        "unlearn.evaluation.eval_mmlu",
        f"--model_path={model_save_path}",
        "--batch_size=8",
    ]
    print(f"Command: {' '.join(mmlu_cmd)}")
    sys.stdout.flush()

    mmlu_result = subprocess.run(
        mmlu_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    print(mmlu_result.stdout)
    if mmlu_result.stderr:
        print(f"STDERR: {mmlu_result.stderr}")

    mmlu_acc = -1.0
    for line in mmlu_result.stdout.split("\n"):
        if "acc" in line.lower() and ":" in line:
            try:
                parts = line.split(":")
                val = float(parts[-1].strip())
                if 0 <= val <= 1:
                    mmlu_acc = val
                    break
            except ValueError:
                continue

    print(f"\n[{datetime.now()}] MMLU accuracy: {mmlu_acc}")
    sys.stdout.flush()

    # Step 4: Record results
    print(f"\n[{datetime.now()}] Recording results...")
    append_to_results(
        str(results_file), args.orth_coef, args.remove_coef, wmdp_acc, mmlu_acc
    )

    print(f"\n[{datetime.now()}] Done!")
    print(
        f"Results: orth_coef={args.orth_coef}, remove_coef={args.remove_coef},"
        f" WMDP={wmdp_acc:.4f}, MMLU={mmlu_acc:.4f}"
    )


if __name__ == "__main__":
    main()
