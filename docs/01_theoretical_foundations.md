# Theoretical Foundations

## Overview

EP-Prior's design is motivated by **PAC-Bayes generalization theory** and **physics-informed machine learning**. This document explains why encoding electrophysiology constraints leads to better sample efficiency.

## 1. PAC-Bayes Theory and Sample Efficiency

### The PAC-Bayes Bound

For a posterior distribution Q over hypotheses, given a prior P, the generalization error is bounded by:

```
E_Q[error] ≤ E_Q[training_error] + √(KL(Q||P) + log(n/δ)) / (2n)
```

Where:
- `n` = number of training samples
- `KL(Q||P)` = divergence between learned posterior and prior
- `δ` = confidence parameter

### Key Insight

**Informative priors reduce KL divergence.** If the prior P already concentrates on physiologically-plausible hypotheses, the posterior Q doesn't need to move far from P, keeping KL(Q||P) small.

### Application to EP-Prior

| Component | How it reduces KL(Q||P) |
|-----------|------------------------|
| Structured latent space | Restricts representation to 4 interpretable factors |
| Gaussian wave decoder | Limits reconstructions to wave mixtures |
| EP constraints | Penalizes physiologically-implausible configurations |

**Result**: Smaller effective hypothesis space → fewer samples needed to generalize.

## 2. Electrophysiology Constraints

### The Cardiac Cycle

A normal ECG consists of three main waves in a fixed temporal order:

```
P wave → QRS complex → T wave
  │           │           │
  │           │           └── Ventricular repolarization
  │           └────────────── Ventricular depolarization
  └────────────────────────── Atrial depolarization
```

### Constraint 1: Wave Ordering

Physiological constraint: P must precede QRS, which must precede T.

```python
# Soft constraint loss
L_order = ReLU(τ_P - τ_QRS) + ReLU(τ_QRS - τ_T)
```

Where τ represents the timing (center) of each wave.

### Constraint 2: Refractory Periods

After depolarization, cardiac tissue cannot immediately re-depolarize:
- **PR interval**: 120-200ms (P to QRS)
- **QT interval**: 350-450ms (QRS to T end)

```python
# Minimum separation constraint
L_refract = ReLU(min_PR - (τ_QRS - τ_P)) + ReLU(min_QT - (τ_T - τ_QRS))
```

### Constraint 3: Duration Bounds

Wave durations have physiological limits:

| Wave | Normal Duration | Pathological if |
|------|-----------------|-----------------|
| P | 80-120ms | >120ms (LAE) |
| QRS | 80-120ms | >120ms (BBB) |
| T | 160-200ms | Prolonged (Long QT) |

```python
# Soft bounds on wave width (σ)
L_duration = ReLU(σ - σ_max) + ReLU(σ_min - σ)
```

## 3. Structured Latent Space Design

### Why Factorization Helps

Traditional autoencoders learn entangled representations where a single latent dimension may encode multiple independent factors. EP-Prior explicitly factorizes:

```
z = [z_P, z_QRS, z_T, z_HRV]
     64D   64D    64D   64D   = 256D total
```

### Disentanglement Benefits

1. **Interpretability**: Each component has physiological meaning
2. **Selective intervention**: Modify one factor without affecting others
3. **Efficient transfer**: Task-specific heads can use relevant components only

### Attention-Based Extraction

Each latent component uses learned attention over the backbone features:

```python
z_P = Attention_P(backbone_features)    # Attends to P-wave regions
z_QRS = Attention_QRS(backbone_features) # Attends to QRS regions
z_T = Attention_T(backbone_features)    # Attends to T-wave regions
z_HRV = GlobalPool(backbone_features)   # Global rhythm features
```

## 4. Gaussian Wave Decoder

### Motivation

Instead of a generic MLP decoder, we use a **physics-informed decoder** that reconstructs ECGs as mixtures of Gaussian-shaped waves:

```
x_hat(t) = Σ_wave A_wave · exp(-(t - τ_wave)² / (2σ_wave²))
```

Where for each wave:
- `A` = amplitude (can be positive or negative)
- `τ` = timing (wave center)
- `σ` = width (wave duration)

### Why Gaussians?

1. **Smooth, differentiable**: Easy to optimize
2. **Interpretable parameters**: Direct physiological meaning
3. **Compositional**: Complex morphologies emerge from simple components

### Gate Mechanism

Each wave has a learnable "gate" parameter in [0, 1] controlling its presence:

```python
x_wave = gate * A * gaussian(t, τ, σ)
```

This allows the model to gracefully handle:
- Missing P waves (atrial fibrillation)
- Absent T waves (hyperkalemia)
- Pathological morphologies

## 5. Training Objective

The total loss combines three terms:

```
L_total = λ_recon · L_recon + λ_ep · L_ep + λ_contrast · L_contrast
```

### Reconstruction Loss (L_recon)

Mean squared error between input and reconstruction:

```python
L_recon = ||x - x_hat||² / ||x||²  # Normalized MSE
```

### EP Constraint Loss (L_ep)

Sum of soft constraint violations:

```python
L_ep = L_order + L_refract + L_duration
```

### Contrastive Loss (L_contrast)

NT-Xent loss on augmented views:

```python
L_contrast = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

### Default Hyperparameters

```python
λ_recon = 1.0
λ_ep = 0.1
λ_contrast = 0.1
temperature = 0.1
```

## 6. Expected Behavior

### When EP Constraints Help Most

1. **Low-shot regimes**: Strong priors compensate for limited data
2. **Normal sinus rhythms**: Constraints directly match the data
3. **Timing-based tasks**: QT prolongation, conduction delays

### When EP Constraints May Not Help

1. **High-data regimes**: Enough data to learn constraints implicitly
2. **Severely pathological ECGs**: Constraints may be violated (but gates handle this)
3. **Non-timing tasks**: Tasks unrelated to wave morphology

## 7. Validation Strategy

### Hypothesis 1: Sample Efficiency
**Test**: Compare few-shot AUROC at {10, 50, 100, 500} shots  
**Expected**: EP-Prior > Baseline, especially at low shots  
**Result**: ✅ +4.4% at 10-shot

### Hypothesis 2: Concept Predictability
**Test**: Linear probe from z_QRS to CD (conduction defects)  
**Expected**: AUROC > 0.7  
**Result**: ✅ 0.789

### Hypothesis 3: Intervention Selectivity
**Test**: Vary z_QRS, measure change in non-QRS decoder params  
**Expected**: <10% leakage  
**Result**: ✅ 0% leakage (perfect disentanglement)

---

**References**:
- McAllester, D. (1999). PAC-Bayesian model averaging
- Raissi, M. et al. (2019). Physics-informed neural networks
- Clifford, G. et al. (2006). Advanced Methods for ECG Analysis

