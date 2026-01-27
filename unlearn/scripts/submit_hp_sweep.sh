#!/bin/bash
# Submit all hyperparameter combinations as separate jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/hp_job.sbatch"

# Hyperparameter grid (defaults: retain=5, remove=23, orth=10)
ORTH_COEFS=(0.0 5.0 10.0 15.0 20.0)
REMOVE_COEFS=(15.0 20.0 23.0 30.0)

echo "Submitting hyperparameter sweep jobs..."
echo "orth_coef values: ${ORTH_COEFS[*]}"
echo "remove_coef values: ${REMOVE_COEFS[*]}"
echo ""

JOB_COUNT=0
for remove_coef in "${REMOVE_COEFS[@]}"; do
    for orth_coef in "${ORTH_COEFS[@]}"; do
        echo "Submitting: orth_coef=$orth_coef, remove_coef=$remove_coef"
        sbatch --export=ORTH_COEF=$orth_coef,REMOVE_COEF=$remove_coef \
               --job-name="hp-o${orth_coef}-r${remove_coef}" \
               "$SBATCH_FILE"
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted $JOB_COUNT jobs"
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: /home/a6a/lucia.a6a/unlearn/runs/tuning_results.md"
