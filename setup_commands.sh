#!/bin/bash
# =============================================================================
# EP-Prior: Complete Setup Script
# =============================================================================
# Run these commands sequentially (some require interaction/waiting)
# =============================================================================

set -e  # Exit on error

PROJECT_ROOT="/root/ep-prior"
cd "$PROJECT_ROOT"

# =============================================================================
# 1. CLONE BASE REPOSITORY (tmehari/ecg-selfsupervised)
# =============================================================================
# This provides: xresnet1d50 backbone, PTB-XL datamodule, training infrastructure

echo ">>> Cloning base repository..."
git clone https://github.com/tmehari/ecg-selfsupervised.git

# =============================================================================
# 2. CLONE REFERENCE REPOSITORIES (for ideas, not direct use)
# =============================================================================
# These are useful for:
# - Architecture ideas
# - Comparing approaches
# - Understanding SOTA

echo ">>> Cloning reference repositories..."
mkdir -p references
cd references

# HuBERT-ECG: Large foundation model, strong baseline comparison
git clone https://github.com/Edoar-do/HuBERT-ECG.git

# CLOCS: Contrastive learning for ECG (patient/temporal invariance)
git clone https://github.com/danikiyasseh/CLOCS.git

# PhysioNet's wfdb tools (for ECG processing utilities)
git clone https://github.com/MIT-LCP/wfdb-python.git

# NeuroKit2: ECG processing, PQRST detection (useful for evaluation)
# (installed via pip, but repo has examples)
git clone https://github.com/neuropsychology/NeuroKit.git

# ecg-kit: MATLAB-based but has good algorithm references for PQRST
# git clone https://github.com/marianux/ecg-kit.git  # Optional, MATLAB

cd "$PROJECT_ROOT"

# =============================================================================
# 3. DOWNLOAD PTB-XL DATASET
# =============================================================================
# PTB-XL: 21,837 12-lead ECGs, 10 seconds, 500Hz
# ~2.5GB download, ~6GB extracted

echo ">>> Setting up data directory..."
mkdir -p data
cd data

# Option A: Direct download (recommended)
echo ">>> Downloading PTB-XL dataset..."
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# Move to cleaner path
mv physionet.org/files/ptb-xl/1.0.3 ptb-xl
rm -rf physionet.org

# Option B: Using PhysioNet credentials (if Option A fails)
# pip install wfdb
# python -c "import wfdb; wfdb.dl_database('ptb-xl', 'ptb-xl')"

cd "$PROJECT_ROOT"

# =============================================================================
# 4. PYTHON ENVIRONMENT SETUP
# =============================================================================

echo ">>> Creating Python environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
pip install \
    pytorch-lightning>=2.0 \
    wandb \
    hydra-core \
    omegaconf \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    wfdb \
    neurokit2 \
    einops \
    rich \
    tqdm

# Install base repo dependencies
cd ecg-selfsupervised
pip install -e .
cd "$PROJECT_ROOT"

# =============================================================================
# 5. CREATE EP-PRIOR PACKAGE STRUCTURE
# =============================================================================

echo ">>> Creating EP-Prior package structure..."

# Main package
mkdir -p ep_prior/{models,losses,data,eval,utils}

# Create __init__.py files
touch ep_prior/__init__.py
touch ep_prior/models/__init__.py
touch ep_prior/losses/__init__.py
touch ep_prior/data/__init__.py
touch ep_prior/eval/__init__.py
touch ep_prior/utils/__init__.py

# Scripts directory
mkdir -p scripts

# Configs directory
mkdir -p configs

# Runs directory (gitignored)
mkdir -p runs

# =============================================================================
# 6. CREATE .gitignore
# =============================================================================

echo ">>> Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.eggs/
*.egg-info/
dist/
build/

# Data (large files)
data/
*.wfdb
*.dat
*.hea

# Runs and logs
runs/
wandb/
lightning_logs/
*.ckpt
*.pt
*.pth

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Reference repos (optional to ignore)
# references/
EOF

# =============================================================================
# 7. VERIFY SETUP
# =============================================================================

echo ">>> Verifying setup..."

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check PTB-XL exists
if [ -d "data/ptb-xl" ]; then
    echo "PTB-XL dataset: OK"
    ls -la data/ptb-xl/ | head -10
else
    echo "WARNING: PTB-XL dataset not found at data/ptb-xl/"
fi

# Check base repo
if [ -d "ecg-selfsupervised" ]; then
    echo "Base repo (ecg-selfsupervised): OK"
else
    echo "WARNING: Base repo not found"
fi

echo ""
echo "==================================================================="
echo "Setup complete! Next steps:"
echo "==================================================================="
echo "1. Activate environment: source venv/bin/activate"
echo "2. Verify PTB-XL: ls data/ptb-xl/"
echo "3. Test base repo: cd ecg-selfsupervised && python -c 'from clinical_ts.xresnet1d import xresnet1d50'"
echo "4. Start implementation in ep_prior/"
echo "==================================================================="

