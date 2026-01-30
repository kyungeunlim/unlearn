#!/usr/bin/env python3
"""Plot forget accuracy over training for checkpoint transfer unlearning."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot forget accuracy over training")
    parser.add_argument("log_file", type=str, help="Path to training log file")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    parser.add_argument(
        "--retain_coef", type=float, default=5, help="Retain coefficient used"
    )
    parser.add_argument(
        "--remove_coef", type=float, default=8, help="Remove coefficient used"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--steps_per_epoch", type=int, default=32, help="Steps per epoch"
    )
    args = parser.parse_args()

    # Extract forget accuracy values from log
    forget_acc = []
    with open(args.log_file, "r") as f:
        for line in f:
            if "forget_argmax_accuracy:" in line:
                val = float(line.strip().split(": ")[1])
                forget_acc.append(val)

    if not forget_acc:
        print(f"No forget_argmax_accuracy found in {args.log_file}")
        return

    # Each data point is logged every epoch
    total_steps = len(forget_acc) * args.steps_per_epoch
    steps = np.arange(0, total_steps, args.steps_per_epoch)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, forget_acc, "b-", alpha=0.7, linewidth=1)

    # Add smoothed line
    window = min(10, len(forget_acc) // 5)
    if window > 1:
        smoothed = np.convolve(forget_acc, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window - 1 :]
        plt.plot(
            smoothed_steps,
            smoothed,
            "r-",
            linewidth=2,
            label=f"Smoothed ({window}-epoch window)",
        )

    plt.axhline(y=0.85, color="g", linestyle="--", label="Target: 85%")
    plt.axhline(
        y=forget_acc[-1],
        color="orange",
        linestyle=":",
        label=f"Final: {forget_acc[-1]*100:.1f}%",
    )

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Forget Accuracy", fontsize=12)
    plt.title(
        (
            f"Checkpoint Transfer with KL Retain "
            f"Loss\nretain_coef={args.retain_coef}, "
            f"remove_coef={args.remove_coef}, "
            f"{args.epochs} epochs ({total_steps} steps)"
        ),
        fontsize=14,
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.70, 0.90)

    # Add hyperparameter text box
    textstr = "\n".join(
        [
            "Hyperparameters:",
            f"retain_coef = {args.retain_coef}",
            f"remove_coef = {args.remove_coef}",
            "lr = 0.001",
            "LoRA rank = 16",
            "batch_size = 32",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # Save plot
    if args.output:
        output_path = args.output
    else:
        output_path = (
            Path(args.log_file).parent.parent
            / "experiment_logs"
            / "forget_acc_over_training.png"
        )

    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print(f"Final forget_acc: {forget_acc[-1]*100:.2f}%")
    print(f"Max forget_acc: {max(forget_acc)*100:.2f}%")
    print(f"Min forget_acc: {min(forget_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
