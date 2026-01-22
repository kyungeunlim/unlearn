#!/bin/bash
#SBATCH --job-name=code_tunnel
#SBATCH --gpus=1            # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=8:00:00
#SBATCH --output=code_tunnel_%j.out
#SBATCH --reservation=aisi

# Start named VS Code tunnel for remote connection to compute node
~/opt/vscode_cli/code tunnel --name "i-ai_compute"
