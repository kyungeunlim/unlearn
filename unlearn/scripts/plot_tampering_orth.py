#!/usr/bin/env python3
"""Plot tampering resistance results for orthogonality circuit breaker models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    # Load results
    results_path = Path(
        "/home/a6a/lucia.a6a/unlearn/runs/tamper_attack_orth100_256/tamper_results_20260128_013124.json"
    )
    with open(results_path) as f:
        results = json.load(f)

    steps = [r["step"] for r in results]
    accs = [r["wmdp_bio_acc"] * 100 for r in results]

    # Model configuration
    model_config = "Orth CB (remove=30, orth=100, retain=2, 256 steps)"
    baseline_wmdp = 42.97  # Original model WMDP Bio
    baseline_mmlu = 42.18  # Post-unlearning MMLU (preserved)
    # unlearned_wmdp = 29.85  # Post-unlearning WMDP Bio

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, "b-o", linewidth=2, markersize=6, label="WMDP Bio Accuracy")

    # Add baselines
    plt.axhline(
        y=baseline_wmdp,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Original Model: {baseline_wmdp:.1f}%",
    )
    plt.axhline(y=25, color="g", linestyle=":", linewidth=2, label="Random Chance: 25%")

    # Add annotations
    plt.annotate(
        f"Start: {accs[0]:.1f}%",
        xy=(steps[0], accs[0]),
        xytext=(steps[0] + 5, accs[0] - 8),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )

    # Find peak
    peak_idx = accs.index(max(accs))
    plt.annotate(
        f"Peak: {accs[peak_idx]:.1f}%",
        xy=(steps[peak_idx], accs[peak_idx]),
        xytext=(steps[peak_idx] + 5, accs[peak_idx] + 5),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )

    plt.annotate(
        f"Final: {accs[-1]:.1f}%",
        xy=(steps[-1], accs[-1]),
        xytext=(steps[-1] - 25, accs[-1] + 8),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )

    plt.xlabel("Finetuning Steps", fontsize=12)
    plt.ylabel("WMDP Bio Accuracy (%)", fontsize=12)
    plt.title(
        f"Tampering Resistance: {model_config}\nFinetune Attack on Bio Data",
        fontsize=14,
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 60)
    plt.xlim(-2, max(steps) + 5)

    # Add recovery info box
    recovery = max(accs) - accs[0]
    exceeds_baseline = max(accs) > baseline_wmdp
    textstr = "\n".join(
        [
            f"Recovery: +{recovery:.1f}% in {steps[peak_idx]} steps",
            (
                f'Exceeds baseline: {"Yes" if exceeds_baseline else "No"} '
                f"({max(accs):.1f}% vs {baseline_wmdp:.1f}%)"
            ),
            f"Post-unlearn MMLU: {baseline_mmlu:.1f}% (preserved)",
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
    output_path = Path(
        "/home/a6a/lucia.a6a/unlearn/experiment_logs/tampering_orth100_256.png"
    )
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print(f"Steps: {steps}")
    print(f"Accuracies: {[f'{a:.2f}%' for a in accs]}")
    print("\nSummary:")
    print(f"  Start: {accs[0]:.2f}%")
    print(f"  Peak: {max(accs):.2f}% at step {steps[peak_idx]}")
    print(f"  Final: {accs[-1]:.2f}%")
    print(f"  Recovery: +{recovery:.2f}%")


if __name__ == "__main__":
    main()
