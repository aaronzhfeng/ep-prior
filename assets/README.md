# EP-Prior Paper Assets

This folder contains figures and tables for the EP-Prior paper.

## Required Figures

1. **figure1_architecture.pdf** - EP-Prior architecture diagram
   - Shows: ECG → Encoder → (z_P, z_QRS, z_T, z_HRV) → Decoder → Reconstructed ECG
   - Include attention pooling visualization
   - Show soft EP constraints in decoder

2. **figure2_sample_efficiency.pdf** - Sample efficiency curves
   - X-axis: Number of training examples (10, 50, 100, 500, Full)
   - Y-axis: AUROC
   - Lines: EP-Prior, PhysioCLR, Generic SSL, Supervised
   - Key message: EP-Prior advantage largest at low-n

3. **figure3_intervention.pdf** - Intervention selectivity test
   - 3 rows: Vary z_P, Vary z_QRS, Vary z_T
   - Show reconstructed ECG waveforms
   - Quantify leakage (% change in non-target components)

4. **figure4_constraints.pdf** - Constraint satisfaction during training
   - Training curves showing decrease in:
     - Ordering violations (τ_P > τ_QRS or τ_QRS > τ_T)
     - PR interval violations
     - σ bound violations

## Required Tables

All tables are currently inline in the paper with placeholder values.
Update after running experiments:

- Table 1: Few-shot classification AUROC
- Table 2: Concept predictability (latent → pathology)
- Table 3: Stratified AUROC by rhythm type
- Table 4: Ablation study

## Generating Figures

Run evaluation scripts to generate figures:

```bash
# After training EP-Prior
python scripts/evaluate_ep_prior.py --mode all --output_dir assets/

# Or individually:
python scripts/evaluate_ep_prior.py --mode fewshot --output_dir assets/
python scripts/evaluate_ep_prior.py --mode intervention --output_dir assets/
python scripts/evaluate_ep_prior.py --mode concept --output_dir assets/
```

## Figure Style Guidelines

- Use consistent color scheme across all figures
- EP-Prior: Blue (#1f77b4)
- Generic SSL: Orange (#ff7f0e)  
- PhysioCLR: Green (#2ca02c)
- Supervised: Red (#d62728)
- Font size: 10pt minimum for readability in two-column format
- Vector format (PDF) preferred for crisp rendering

