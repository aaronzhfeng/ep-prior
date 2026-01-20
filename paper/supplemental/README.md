# EP-Prior: Supplemental Code

This supplemental material contains the core implementation of **EP-Prior**, a self-supervised framework for interpretable ECG representation learning.

## Structure

```
supplemental/
├── ep_prior/
│   ├── models/
│   │   ├── structured_encoder.py      # Attention-pooled encoder (§4.2)
│   │   ├── gaussian_wave_decoder.py   # EP-constrained decoder (§4.3)
│   │   └── lightning_module.py        # Training module
│   ├── losses/
│   │   └── ep_constraints.py          # Soft EP constraint losses (§4.4, Eq 8-11)
│   ├── data/
│   │   └── ptbxl_dataset.py           # PTB-XL dataset loading
│   └── eval/
│       ├── fewshot.py                 # Few-shot classification
│       └── intervention.py            # Intervention selectivity tests
└── scripts/
    ├── train_ep_prior.py              # Training script
    └── evaluate_ep_prior.py           # Evaluation script
```

## Core Components

### 1. Structured Encoder (`structured_encoder.py`)
- **StructuredEncoder**: Maps 12-lead ECG to structured latents $(z_P, z_{QRS}, z_T, z_{HRV})$
- Uses attention pooling to learn wave-specific temporal regions
- Backbone: xresnet1d50 from [Mehari & Strodthoff, 2022]

### 2. Gaussian Wave Decoder (`gaussian_wave_decoder.py`)
- **GaussianWaveDecoder**: Reconstructs ECG from structured latents
- Model: $\hat{x}_t = \sum_{w \in \{P,QRS,T\}} g_w \cdot A_w \cdot \exp(-(t-\tau_w)^2 / 2\sigma_w^2)$
- Shared timing $(\tau_w, \sigma_w)$ across leads; per-lead amplitudes $A_w$
- QRS uses K=3 mixture for Q/R/S morphology

### 3. EP Constraint Losses (`ep_constraints.py`)
- **Ordering**: $\mathcal{L}_{order} = \text{softplus}(\tau_P - \tau_{QRS}) + \text{softplus}(\tau_{QRS} - \tau_T)$
- **Refractory**: Minimum PR and QT intervals
- **Duration bounds**: Wave width constraints
- All constraints gated by wave presence for pathological cases

## Requirements

```
torch>=1.10
pytorch-lightning>=1.6
numpy
pandas
wfdb  # For PTB-XL loading
```

## Usage

### Training
```bash
python scripts/train_ep_prior.py --data_path /path/to/ptb-xl --batch_size 64
```

### Evaluation
```bash
python scripts/evaluate_ep_prior.py --checkpoint /path/to/model.ckpt
```

## Dataset

Experiments use [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), a large publicly available 12-lead ECG dataset with 21,837 records.

## License

This code is provided for academic research purposes. See main paper for citation information.

