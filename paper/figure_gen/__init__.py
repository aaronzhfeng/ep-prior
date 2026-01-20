"""
EP-Prior Paper Figure Generation

Individual figure scripts:
- fig2_sample_efficiency.py: Sample efficiency curves
- fig3_intervention.py: Intervention selectivity heatmap  
- fig4_ablation.py: Ablation study bar chart
- fig5_tsne.py: t-SNE visualization (vertical layout)
- fig6_reconstruction.py: ECG reconstruction examples

Orchestrator:
- generate_all.py: Generate all figures at once
"""

from .common import setup_style, save_figure, COLORS, RESULTS_DIR, OUTPUT_DIR

