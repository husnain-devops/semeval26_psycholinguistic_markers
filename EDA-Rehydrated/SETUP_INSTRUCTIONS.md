# Virtual Environment Setup Instructions

## Quick Setup (Recommended)

Run the automated script to recreate the virtual environment with all NumPy 2.x compatible packages:

```bash
cd /home/husnain/semeval26_psycholinguistic_markers
bash EDA-Rehydrated/recreate_venv.sh
```

## Manual Setup

If you prefer to set up manually, follow these steps:

### 1. Remove old virtual environment

```bash
cd /home/husnain/semeval26_psycholinguistic_markers
rm -rf .venv
```

### 2. Create new virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate virtual environment

```bash
source .venv/bin/activate
```

### 4. Upgrade pip and install PyTorch (NumPy 2.x compatible)

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install all other requirements

```bash
cd EDA-Rehydrated
pip install -r requirements.txt
```

### 6. Install spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 7. Set up Jupyter kernel (for notebooks)

```bash
pip install ipykernel
python -m ipykernel install --user --name=semeval26_env --display-name "Python (semeval26_env)"
```

## Verify Installation

After setup, verify the installation:

```bash
python -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}'); tensor = torch.tensor([1,2,3]); print(f'Compatibility: {tensor.numpy()}')"
```

You should see:
- NumPy version starting with `2.`
- PyTorch version (latest)
- A numpy array `[1 2 3]` without errors

## Using with Jupyter Notebooks

1. Activate the environment: `source .venv/bin/activate`
2. Start Jupyter: `jupyter notebook` or `jupyter lab`
3. Select the kernel: `semeval26_env` (or the kernel name you specified)

## Troubleshooting

### NumPy 2.x Compatibility Issues

If you see errors like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x":

1. Make sure you're using the latest PyTorch:
   ```bash
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. Restart your Jupyter kernel after installing/upgrading packages

3. If issues persist, try installing PyTorch from the nightly build:
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

### Package Installation Issues

If some packages fail to install:

1. Make sure you have the latest pip: `pip install --upgrade pip`
2. Install system dependencies if needed (varies by OS)
3. Try installing packages one by one to identify the problematic package

## Notes

- The virtual environment is located at: `.venv/` in the project root
- All packages are configured for NumPy 2.x compatibility
- PyTorch is installed separately to ensure NumPy 2.x compatibility
- The `recreate_venv.sh` script automates all these steps

