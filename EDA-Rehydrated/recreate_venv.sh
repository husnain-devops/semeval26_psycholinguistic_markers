#!/bin/bash
# Script to recreate virtual environment with NumPy 2.x compatible packages

echo "=========================================="
echo "Recreating Virtual Environment"
echo "=========================================="

# Navigate to the project root (where .venv is located)
cd "$(dirname "$0")/.." || exit 1

# Remove old virtual environment
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
    echo "✓ Old .venv removed"
else
    echo "No existing .venv found"
fi

# Create new virtual environment
echo ""
echo "Creating new virtual environment..."
python3 -m venv .venv
echo "✓ New .venv created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch first (with NumPy 2.x compatible version)
echo ""
echo "Installing PyTorch (NumPy 2.x compatible)..."
# Install latest PyTorch which should support NumPy 2.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other requirements
echo ""
echo "Installing all requirements..."
cd EDA-Rehydrated || exit 1
pip install -r requirements.txt

# Install spaCy English model
echo ""
echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm

echo ""
echo "=========================================="
echo "✓ Virtual environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "For Jupyter notebooks, make sure to:"
echo "  1. Activate the environment"
echo "  2. Install ipykernel: pip install ipykernel"
echo "  3. Register kernel: python -m ipykernel install --user --name=semeval26_env"
echo "  4. Select the kernel in your notebook"
echo ""

