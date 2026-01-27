#!/bin/bash
# Script to set up Jupyter kernel for the filter conda environment

# Activate the base conda environment
source ~/miniforge3/bin/activate

# Check if filter environment exists
if ! conda env list | grep -q "^filter "; then
    echo "Error: 'filter' conda environment not found!"
    echo "Please create it first or check the environment name."
    exit 1
fi

# Activate the filter environment
conda activate filter

# Check if jupyter and ipykernel are installed
echo "Checking for Jupyter and ipykernel in filter environment..."
if ! python -c "import jupyter" 2>/dev/null; then
    echo "Installing jupyter in filter environment..."
    conda install -y jupyter
fi

if ! python -c "import ipykernel" 2>/dev/null; then
    echo "Installing ipykernel in filter environment..."
    conda install -y ipykernel
fi

# Install the kernel spec
echo "Installing kernel spec for filter environment..."
python -m ipykernel install --user \
    --name "filter" \
    --display-name "Python (filter - Filtering for Danger)"

echo "=========================================="
echo "Kernel installation complete!"
echo "The 'Python (filter - Filtering for Danger)' kernel is now available in Jupyter."
echo ""
echo "To launch Jupyter with this environment, run:"
echo "  sbatch launch_jupyter_filter.sh"
echo "=========================================="

# Deactivate the environment
conda deactivate