#!/usr/bin/env python3
"""Plot tampering resistance (finetune attack) results."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot tampering resistance results")
    parser.add_argument("log_file", type=str, help="Path to finetune attack log file")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    parser.add_argument("--model_name", type=str, default="Lens Unlearned Model", help="Model name for title")
    parser.add_argument("--baseline", type=float, default=0.4297, help="Original model WMDP Bio accuracy")
    args = parser.parse_args()

    # Extract results from log
    steps = []
    accs = []

    with open(args.log_file, 'r') as f:
        content = f.read()

    # Find all evaluation results
    pattern = r"Step (\d+): Evaluation Results: \{'wmdp_bio_robust': ([0-9.]+)"
    matches = re.findall(pattern, content)

    for step, acc in matches:
        steps.append(int(step))
        accs.append(float(acc) * 100)  # Convert to percentage

    if not steps:
        print(f"No evaluation results found in {args.log_file}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, 'b-o', linewidth=2, markersize=8, label='WMDP Bio Accuracy')

    # Add baseline
    plt.axhline(y=args.baseline * 100, color='r', linestyle='--', linewidth=2,
                label=f'Original Model: {args.baseline*100:.1f}%')
    plt.axhline(y=25, color='g', linestyle=':', linewidth=2, label='Random Chance: 25%')

    # Add annotations
    plt.annotate(f'Start: {accs[0]:.1f}%', xy=(steps[0], accs[0]),
                 xytext=(steps[0]+2, accs[0]-5), fontsize=10)
    if len(accs) > 1:
        plt.annotate(f'After attack: {accs[-1]:.1f}%', xy=(steps[-1], accs[-1]),
                     xytext=(steps[-1]-5, accs[-1]+3), fontsize=10)

    plt.xlabel('Finetuning Steps', fontsize=12)
    plt.ylabel('WMDP Bio Accuracy (%)', fontsize=12)
    plt.title(f'Tampering Resistance: {args.model_name}\nFinetune Attack on Bio Data', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 60)

    # Add recovery info
    if len(accs) > 1:
        recovery = accs[-1] - accs[0]
        textstr = f'Recovery: +{recovery:.1f}% in {steps[-1]} steps'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.log_file).parent.parent / 'experiment_logs' / 'tampering_lens.png'

    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print(f"Steps: {steps}")
    print(f"Accuracies: {[f'{a:.2f}%' for a in accs]}")


if __name__ == "__main__":
    main()
