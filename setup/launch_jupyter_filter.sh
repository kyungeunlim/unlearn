#!/bin/bash
#SBATCH --job-name=jupyter_filter_env
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4  # Full node with 4 GPUs
#SBATCH --time=08:00:00  # 8 hours for analysis work
#SBATCH --reservation=aisi

# Activate the filter conda environment
source ~/miniforge3/bin/activate filter

# Add pre-installed kernelspecs to the Jupyter data search path
export JUPYTER_PATH="/tools/brics/jupyter/jupyter_data${JUPYTER_PATH:+:}${JUPYTER_PATH:-}"

# Get the HSN address for the compute node
HSN_FQDN="$(hostname).hsn.ai-p1.isambard.ac.uk"
LISTEN_IP=$(dig "${HSN_FQDN}" A +short | tail -n 1)
LISTEN_PORT=8888

# Print connection information
echo "=========================================="
echo "Jupyter server starting on ${LISTEN_IP}:${LISTEN_PORT}"
echo "To connect from your local machine, run:"
echo "ssh -T -L localhost:8888:${LISTEN_IP}:${LISTEN_PORT} <PROJECT>.aip1.isambard"
echo "Then open http://localhost:8888 in your browser"
echo "=========================================="

# Start JupyterLab
set -o xtrace
jupyter lab --no-browser --ip="${LISTEN_IP}" --port="${LISTEN_PORT}"