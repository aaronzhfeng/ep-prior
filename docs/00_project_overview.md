# EP-Prior: Project Overview

## What is EP-Prior?

**EP-Prior** (Electrophysiology-Prior) is a self-supervised learning framework for ECG representation learning that achieves superior sample efficiency through physics-informed architectural constraints. Instead of treating ECGs as generic time series, EP-Prior encodes domain knowledge about cardiac electrophysiology directly into the model architecture.

## The Problem

Standard self-supervised learning (SSL) methods for ECGs treat them as arbitrary 1D signals, ignoring decades of cardiology knowledge. This leads to:

1. **Poor sample efficiency**: Need thousands of labeled examples for fine-tuning
2. **Uninterpretable representations**: Latent spaces have no physiological meaning
3. **Limited generalization**: Models don't transfer well across ECG tasks

## Our Solution: Structured Latents + EP Constraints

EP-Prior introduces two key innovations:

### 1. Structured Latent Space
Instead of a single latent vector, we decompose ECG representations into:
- **z_P**: P-wave characteristics (atrial depolarization)
- **z_QRS**: QRS complex characteristics (ventricular depolarization)
- **z_T**: T-wave characteristics (ventricular repolarization)
- **z_HRV**: Heart rate variability / rhythm features

### 2. EP-Constrained Decoder
A Gaussian wave state-space model that reconstructs ECGs as a mixture of physiologically-plausible waves, with soft constraints enforcing:
- **Wave ordering**: P before QRS before T
- **Refractory periods**: Minimum time between waves
- **Duration bounds**: Physiological limits on wave widths

## Key Claims & Results

| Claim | Target | Achieved | Status |
|-------|--------|----------|--------|
| Few-shot improvement at 10-shot | EP-Prior > Baseline | **+7.2% AUROC** | ✅ |
| Per-condition improvement | EP-Prior > Baseline (all) | **5/5 conditions** | ✅ |
| EP constraints essential | No-EP < Full EP-Prior | **-18% at 10-shot** | ✅ |
| Ablation vs Baseline | No-EP < Baseline | **-10.8%** | ✅ |

### Sample Efficiency Curve (Main Result)

```
Shots  | EP-Prior | Baseline | Delta
-------|----------|----------|-------
10     | 0.699    | 0.627    | +7.2%  ← Largest gain
50     | 0.790    | 0.739    | +5.1%
100    | 0.805    | 0.766    | +3.9%
500    | 0.826    | 0.812    | +1.4%
```

### Ablation Study (Critical Finding)

```
Model              | 10-shot | 500-shot
-------------------|---------|----------
EP-Prior (Full)    | 0.699   | 0.826
Baseline           | 0.627   | 0.812
EP-Prior (No EP)   | 0.519 ❌ | 0.650 ❌
```

**Key insight**: EP-Prior's advantage is largest when labeled data is scarce (10-shot). The ablation proves **EP constraints are essential** - structured latents alone perform worse than baseline!

## Project Structure

```
ep-prior/
├── ep_prior/                 # Main package
│   ├── models/               # Model architectures
│   ├── losses/               # Loss functions
│   ├── data/                 # Data loading
│   └── eval/                 # Evaluation modules
├── scripts/                  # Training and evaluation scripts
├── runs/                     # Checkpoints and results
├── data/                     # PTB-XL dataset
└── docs/                     # This documentation
```

## Quick Start

```bash
# Environment setup
cd /root/ep-prior
source venv/bin/activate

# All results already computed! See:
ls results/                    # Data files
ls results/figures/            # Paper figures
cat results/results_summary.json  # All key numbers

# To regenerate figures (optional):
python scripts/generate_paper_figures.py  # Auto-discovers latest results
```

## Theoretical Foundation

EP-Prior is motivated by **PAC-Bayes generalization theory**, which states that models with stronger (more informative) priors require fewer samples to generalize. By encoding electrophysiology knowledge as architectural constraints, we effectively narrow the hypothesis space to physiologically-plausible representations.

See [01_theoretical_foundations.md](./01_theoretical_foundations.md) for detailed theory.

## Next Steps

For a complete guide on what to do next, see [08_next_steps.md](./08_next_steps.md).

---

**Last Updated**: January 19, 2026  
**Status**: ✅ All experiments complete. Results in `results/`. Ready for paper writing.

