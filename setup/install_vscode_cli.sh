#!/bin/bash
# Script to install VS Code CLI on Isambard-AI (ARM64 architecture)

echo "=========================================="
echo "VS Code CLI Installation Script"
echo "=========================================="

# Check if VS Code CLI is already installed
if [ -f ~/opt/vscode_cli/code ]; then
    echo "VS Code CLI is already installed at ~/opt/vscode_cli/code"
    echo "Current version:"
    ~/opt/vscode_cli/code --version
    echo ""
    read -p "Do you want to reinstall/update? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Create installation directory
echo "Creating installation directory..."
mkdir -p ~/opt/vscode_cli

# Download VS Code CLI for ARM64
echo "Downloading VS Code CLI for ARM64 architecture..."
curl --location --output /tmp/vscode_cli.tar.gz "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download VS Code CLI"
    exit 1
fi

# Extract the archive
echo "Extracting VS Code CLI..."
tar -C ~/opt/vscode_cli --extract --file /tmp/vscode_cli.tar.gz

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract VS Code CLI"
    exit 1
fi

# Clean up
rm /tmp/vscode_cli.tar.gz

# Verify installation
echo "Verifying installation..."
if [ -f ~/opt/vscode_cli/code ]; then
    echo "VS Code CLI successfully installed!"
    echo "Version:"
    ~/opt/vscode_cli/code --version
    echo ""
    echo "=========================================="
    echo "Installation complete!"
    echo ""
    echo "To start a VS Code tunnel on a compute node:"
    echo "  sbatch launch_vscode_tunnel.sh"
    echo ""
    echo "The VS Code CLI is installed at:"
    echo "  ~/opt/vscode_cli/code"
    echo "=========================================="
else
    echo "Error: Installation verification failed"
    exit 1
fi