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
| Few-shot improvement at 10-shot | EP-Prior > Baseline | +4.4% AUROC | ✅ |
| Concept predictability (z_QRS → CD) | >0.7 AUROC | 0.789 | ✅ |
| Concept predictability (z_T → STTC) | >0.7 AUROC | 0.883 | ✅ |
| Intervention selectivity | <10% leakage | 0% | ✅ |

### Sample Efficiency Curve (Main Result)

```
Shots  | EP-Prior | Baseline | Delta
-------|----------|----------|-------
10     | 0.726    | 0.682    | +4.4%
50     | 0.801    | 0.765    | +3.6%
100    | 0.814    | 0.793    | +2.1%
500    | 0.826    | 0.811    | +1.5%
```

**Key insight**: EP-Prior's advantage is largest when labeled data is scarce (10-shot), exactly as predicted by PAC-Bayes theory. The EP constraints act as informative priors that reduce the effective hypothesis space.

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

# Run full evaluation
python scripts/run_full_evaluation.py --num_seeds 3

# Generate paper figures
python scripts/generate_paper_figures.py --results_dir runs/evaluation_20260118_173518
```

## Theoretical Foundation

EP-Prior is motivated by **PAC-Bayes generalization theory**, which states that models with stronger (more informative) priors require fewer samples to generalize. By encoding electrophysiology knowledge as architectural constraints, we effectively narrow the hypothesis space to physiologically-plausible representations.

See [01_theoretical_foundations.md](./01_theoretical_foundations.md) for detailed theory.

## Next Steps

For a complete guide on what to do next, see [08_next_steps.md](./08_next_steps.md).

---

**Last Updated**: January 18, 2026  
**Status**: Core experiments complete, paper figures ready

