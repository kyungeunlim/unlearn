#!/bin/bash
#SBATCH --job-name=vscode_tunnel_filter
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4  # Full node with 4 GPUs (288 CPU cores, 460GB memory)
#SBATCH --time=08:00:00    # 8 hours for development work
#SBATCH --output=vscode_tunnel_%j.out

source ~/miniforge3/bin/activate snake

# Set environment variables for GPU usage
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Display node information
echo "=========================================="
echo "VS Code Tunnel Starting on Node: $(hostname)"
echo "GPUs Available: 4 x GH200 (96GB each)"
echo "CPU Cores: 288 (72 per GPU)"
echo "Total Memory: ~460GB"
echo "Working Directory: $(pwd)"
echo "Conda Environment: filter"
echo "=========================================="

# Check if VS Code CLI is installed
if [ ! -f ~/opt/vscode_cli/code ]; then
    echo "Error: VS Code CLI not found at ~/opt/vscode_cli/code"
    echo "Please run ./install_vscode_cli.sh first"
    exit 1
fi

# Start named VS Code tunnel for remote connection to compute node
# Using a unique name that includes the project for easy identification
echo "Starting VS Code tunnel with name: a5k-filter-compute"
echo "Please check the output below for the authentication code and connection URL"
echo "=========================================="

~/opt/vscode_cli/code tunnel --name "a5k-filter-compute" --accept-server-license-terms