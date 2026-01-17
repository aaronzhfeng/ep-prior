# EP-Prior: Interpretable ECG Representations via Electrophysiology Constraints

**Target:** IJCAI 2026 (AI and Health Track)  
**Paper deadline:** January 19, 2026

---

## Quick Start

```bash
cd /root/ep-prior
source venv/bin/activate

# Quick test (CPU)
CUDA_VISIBLE_DEVICES="" python scripts/train_ep_prior.py \
    --fast_dev_run --batch_size 4 --num_workers 0

# Full training
python scripts/train_ep_prior.py \
    --max_epochs 100 --batch_size 64

# Evaluation (after training)
python scripts/evaluate_ep_prior.py \
    --checkpoint runs/your_experiment/checkpoints/last.ckpt
```

---

## Project Structure

```
ep-prior/
├── ep_prior/                    # Core implementation (~2,500 lines)
│   ├── models/
│   │   ├── gaussian_wave_decoder.py   # P/QRS/T Gaussian reconstruction
│   │   ├── structured_encoder.py      # Attention-pooled encoder
│   │   └── lightning_module.py        # Training loop
│   ├── losses/
│   │   └── ep_constraints.py          # Ordering, refractory, duration
│   ├── data/
│   │   └── ptbxl_dataset.py           # PTB-XL loader + augmentations
│   └── eval/
│       ├── probes.py                  # Linear probe utilities
│       ├── fewshot.py                 # Few-shot classification
│       ├── intervention.py            # Latent intervention tests
│       └── concept_predictability.py  # z_QRS → BBB validation
├── scripts/
│   ├── train_ep_prior.py              # Training entrypoint
│   └── evaluate_ep_prior.py           # Evaluation entrypoint
├── ecg-selfsupervised/                # Base repo (xresnet1d50)
└── data/ptb-xl/                       # PTB-XL dataset (17K+ training)
```

---

## Architecture

| Component | Description | Params |
|-----------|-------------|--------|
| **Encoder** | xresnet1d50 + attention pooling | 1.2M |
| **Decoder** | Gaussian wave model (P/QRS/T) | 250K |
| **Total** | | **1.5M** |

**Latent Structure:**
- `z_P`: 32 dims (P-wave / atrial activity)
- `z_QRS`: 128 dims (QRS / ventricular depolarization)
- `z_T`: 64 dims (T-wave / repolarization)
- `z_HRV`: 32 dims (heart rate variability)

---

## Key Features

### EP-Constrained Decoder
```
x̂_t = Σ_{w∈{P,QRS,T}} A_w · exp(-(t-τ_w)²/2σ_w²)
```
- Timing τ, width σ, amplitude A per wave
- QRS uses K=3 mixture (Q/R/S components)
- Presence gates for pathological cases (AFib, BBB)

### Soft EP Constraints
- **Ordering:** τ_P < τ_QRS < τ_T
- **Refractory:** |τ_QRS - τ_P| > ΔPR_min
- **Duration:** σ ∈ [σ_min, σ_max]

---

## Evaluation Suite

### 1. Few-Shot Classification
Tests PAC-Bayes prediction: EP-Prior gains largest at low-n
```python
python scripts/evaluate_ep_prior.py --checkpoint model.ckpt
# Outputs: AUROC @ {10, 50, 100, 500} shots
```

### 2. Intervention Selectivity
Vary z_QRS → only QRS changes (target: <10% leakage)
```python
from ep_prior.eval import run_intervention_evaluation
results = run_intervention_evaluation(model, dataloader)
```

### 3. Concept Predictability
z_QRS predicts CD (Conduction Disturbance), z_T predicts STTC
```python
from ep_prior.eval import run_concept_evaluation
results, summary = run_concept_evaluation(model, train_ds, test_ds)
```

---

## Training

```bash
# Basic training
python scripts/train_ep_prior.py \
    --max_epochs 100 \
    --batch_size 64 \
    --lam_ep 0.5

# With contrastive learning
python scripts/train_ep_prior.py \
    --lam_contrast 0.1 \
    --max_epochs 100

# With W&B logging
python scripts/train_ep_prior.py \
    --use_wandb \
    --wandb_project ep-prior \
    --experiment_name my_run
```

---

## Implementation Status

- [x] GaussianWaveDecoder (344 lines)
- [x] EP constraint losses (327 lines)
- [x] StructuredEncoder (402 lines)
- [x] LightningModule (448 lines)
- [x] PTB-XL data pipeline (338 lines)
- [x] Few-shot evaluation
- [x] Intervention tests
- [x] Concept predictability
- [ ] Run full training
- [ ] Generate paper figures

---

## Dataset: PTB-XL

- **Records:** 21,837 12-lead ECGs
- **Duration:** 10 seconds @ 100Hz
- **Labels:** 71 SCP-ECG diagnostic statements
- **Superclasses:** NORM, MI, STTC, CD, HYP
- **Splits:** Folds 1-8 train, 9 val, 10 test

---

## Reference Repositories

| Repo | Purpose |
|------|---------|
| **tmehari/ecg-selfsupervised** | xresnet1d50, PTB-XL loading |
| **Edoar-do/HuBERT-ECG** | Large foundation model comparison |
| **danikiyasseh/CLOCS** | Contrastive ECG ideas |
| **neuropsychology/NeuroKit** | PQRST detection utilities |
