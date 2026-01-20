#!/usr/bin/env python
"""
Generate data-based figures (no model required).
Only generates figures 2 and 4 which read from CSV files.
"""

import fig2_sample_efficiency
import fig4_ablation

if __name__ == "__main__":
    print("=" * 60)
    print("Generating data-based figures (no model required)")
    print("=" * 60)
    
    fig2_sample_efficiency.generate_figure()
    fig4_ablation.generate_figure()
    
    print("\n" + "=" * 60)
    print("Done! Check paper/figures/ for output.")
    print("=" * 60)

