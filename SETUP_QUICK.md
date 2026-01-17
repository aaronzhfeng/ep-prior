# Quick Setup Commands

Run these in order:

```bash
cd /root/ep-prior

# =============================================================================
# 1. CLONE REPOS
# =============================================================================

# Base repo (REQUIRED)
git clone https://github.com/tmehari/ecg-selfsupervised.git

# Reference repos (OPTIONAL but useful)
mkdir -p references && cd references
git clone https://github.com/Edoar-do/HuBERT-ECG.git
git clone https://github.com/danikiyasseh/CLOCS.git
git clone https://github.com/neuropsychology/NeuroKit.git
cd ..

# =============================================================================
# 2. DOWNLOAD PTB-XL DATASET (~2.5GB)
# =============================================================================

mkdir -p data && cd data
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
mv physionet.org/files/ptb-xl/1.0.3 ptb-xl && rm -rf physionet.org
cd ..

# =============================================================================
# 3. PYTHON ENVIRONMENT
# =============================================================================

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install pytorch-lightning wandb hydra-core omegaconf numpy pandas scipy \
    scikit-learn matplotlib seaborn wfdb neurokit2 einops rich tqdm

# Install base repo
cd ecg-selfsupervised && pip install -e . && cd ..

# =============================================================================
# 4. CREATE PACKAGE STRUCTURE
# =============================================================================

mkdir -p ep_prior/{models,losses,data,eval,utils}
mkdir -p scripts configs runs
touch ep_prior/__init__.py ep_prior/{models,losses,data,eval,utils}/__init__.py

# =============================================================================
# 5. VERIFY
# =============================================================================

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from clinical_ts.xresnet1d import xresnet1d50; print('xresnet1d50: OK')"
ls data/ptb-xl/
```

---

## Reference Repos Summary

| Repo | URL | Why Useful |
|------|-----|------------|
| ecg-selfsupervised | github.com/tmehari/ecg-selfsupervised | **BASE**: xresnet1d50, PTB-XL loader, CPC/SimCLR/BYOL |
| HuBERT-ECG | github.com/Edoar-do/HuBERT-ECG | Large foundation model, baseline comparison |
| CLOCS | github.com/danikiyasseh/CLOCS | Contrastive ECG ideas |
| NeuroKit | github.com/neuropsychology/NeuroKit | PQRST detection for evaluation |

---

## After Setup

Start implementation:
```bash
source venv/bin/activate
# Begin with: ep_prior/models/gaussian_wave_decoder.py
```

