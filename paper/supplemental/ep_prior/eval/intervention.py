"""
Intervention Tests for EP-Prior Interpretability

Tests whether the structured latent space is truly disentangled:
1. Vary z_QRS while holding z_P, z_T fixed → only QRS should change
2. Measure "leakage" to other components

This is the KEY differentiator from post-hoc explanations:
"We can selectively modify one component and observe isolated effects"
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


class InterventionTester:
    """
    Tests intervention selectivity of EP-Prior decoder.
    
    For each wave component, varies the corresponding latent along
    a direction (e.g., principal axis) and measures:
    1. Parameter changes in target component (should be large)
    2. Parameter changes in other components (should be small = low leakage)
    """
    
    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: Trained EPPriorSSL model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
    
    def _get_baseline_latents(
        self,
        dataloader,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Get baseline latents from data for reference."""
        all_z = {"P": [], "QRS": [], "T": [], "HRV": []}
        
        with torch.no_grad():
            count = 0
            for batch in tqdm(dataloader, desc="Extracting baseline latents"):
                if count >= n_samples:
                    break
                x = batch["x"].to(self.device)
                z, _ = self.model.encoder(x, return_attention=False)
                
                for key in all_z:
                    all_z[key].append(z[key])
                count += x.shape[0]
        
        return {k: torch.cat(v, dim=0)[:n_samples] for k, v in all_z.items()}
    
    def _compute_principal_direction(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute first principal component direction."""
        # Center
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # SVD
        U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
        
        # First principal direction
        return Vh[0]  # (d,)
    
    def run_intervention(
        self,
        base_z: Dict[str, torch.Tensor],
        target_component: str,
        direction: torch.Tensor,
        n_steps: int = 11,
        scale: float = 3.0,
        T: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Run intervention by varying one component along a direction.
        
        Args:
            base_z: Baseline latent dict (single sample)
            target_component: "P", "QRS", or "T"
            direction: Direction to vary (d,)
            n_steps: Number of interpolation steps
            scale: Range of variation in std units
            T: Time steps for decoding
            
        Returns:
            Dict with:
            - "params": Decoder params at each step
            - "reconstructions": Decoded waveforms
        """
        alphas = torch.linspace(-scale, scale, n_steps).to(self.device)
        
        all_params = []
        all_recons = []
        
        with torch.no_grad():
            for alpha in alphas:
                # Create modified latent
                z_modified = {k: v.clone() for k, v in base_z.items()}
                z_modified[target_component] = (
                    base_z[target_component] + alpha * direction.unsqueeze(0)
                )
                
                # Decode (only P, QRS, T)
                z_decode = {k: z_modified[k] for k in ["P", "QRS", "T"]}
                x_hat, params = self.model.decoder(z_decode, T=T)
                
                all_params.append(params)
                all_recons.append(x_hat)
        
        # Stack params
        stacked_params = {}
        for wave in ["P", "QRS", "T"]:
            stacked_params[wave] = {
                "tau": torch.stack([p[wave]["tau"] for p in all_params]),
                "sig": torch.stack([p[wave]["sig"] for p in all_params]),
                "gate": torch.stack([p[wave]["gate"] for p in all_params]),
            }
        
        return {
            "params": stacked_params,
            "reconstructions": torch.stack(all_recons),
            "alphas": alphas,
        }
    
    def compute_leakage(
        self,
        intervention_results: Dict,
        target_component: str,
    ) -> Dict[str, float]:
        """
        Compute leakage metrics from intervention.
        
        Leakage = change in non-target params / change in target params
        
        Low leakage (<10%) indicates good disentanglement.
        """
        params = intervention_results["params"]
        
        # Compute changes for each wave
        changes = {}
        for wave in ["P", "QRS", "T"]:
            tau_range = params[wave]["tau"].max() - params[wave]["tau"].min()
            sig_range = params[wave]["sig"].max() - params[wave]["sig"].min()
            changes[wave] = {
                "tau": tau_range.item(),
                "sig": sig_range.item(),
            }
        
        # Compute leakage ratios
        target_tau = changes[target_component]["tau"]
        target_sig = changes[target_component]["sig"]
        
        leakage = {}
        for wave in ["P", "QRS", "T"]:
            if wave == target_component:
                continue
            
            tau_leak = changes[wave]["tau"] / (target_tau + 1e-8)
            sig_leak = changes[wave]["sig"] / (target_sig + 1e-8)
            leakage[f"{wave}_tau_leakage"] = tau_leak
            leakage[f"{wave}_sig_leakage"] = sig_leak
        
        # Summary
        all_leakages = list(leakage.values())
        leakage["mean_leakage"] = np.mean(all_leakages)
        leakage["max_leakage"] = np.max(all_leakages)
        
        return leakage
    
    def evaluate_all_interventions(
        self,
        dataloader,
        n_base_samples: int = 10,
        n_steps: int = 11,
        scale: float = 2.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run interventions for all components and compute average leakage.
        
        Args:
            dataloader: DataLoader to get baseline samples
            n_base_samples: Number of base samples to average over
            n_steps: Interpolation steps
            scale: Variation scale
            
        Returns:
            Dict with leakage metrics per target component
        """
        # Get baseline latents
        baseline_z = self._get_baseline_latents(dataloader, n_base_samples)
        
        results = {}
        
        for target in ["P", "QRS", "T"]:
            print(f"Testing {target} intervention...")
            
            # Get direction (PC1 of target latent)
            direction = self._compute_principal_direction(baseline_z[target])
            
            all_leakages = []
            
            for i in range(n_base_samples):
                # Get single sample
                base_z = {k: v[i:i+1] for k, v in baseline_z.items()}
                
                # Run intervention
                interv_results = self.run_intervention(
                    base_z, target, direction, n_steps, scale
                )
                
                # Compute leakage
                leakage = self.compute_leakage(interv_results, target)
                all_leakages.append(leakage)
            
            # Average leakages
            avg_leakage = {}
            for key in all_leakages[0]:
                avg_leakage[key] = np.mean([l[key] for l in all_leakages])
            
            results[target] = avg_leakage
        
        return results
    
    def visualize_intervention(
        self,
        intervention_results: Dict,
        target_component: str,
        lead_idx: int = 1,  # Lead II
        save_path: Optional[str] = None,
    ):
        """
        Visualize intervention effects on waveform reconstruction.
        """
        recons = intervention_results["reconstructions"]  # (n_steps, 1, 12, T)
        alphas = intervention_results["alphas"]
        params = intervention_results["params"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Waveform evolution
        ax = axes[0, 0]
        n_steps = recons.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
        
        for i in range(n_steps):
            ax.plot(
                recons[i, 0, lead_idx].cpu().numpy(),
                color=colors[i],
                alpha=0.7,
                label=f"α={alphas[i]:.1f}" if i % 3 == 0 else None,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Varying z_{target_component} (Lead {lead_idx})")
        ax.legend()
        
        # Plot 2-4: Parameter trajectories
        for idx, wave in enumerate(["P", "QRS", "T"]):
            ax = axes.flat[idx + 1]
            
            tau = params[wave]["tau"].squeeze().cpu().numpy()
            sig = params[wave]["sig"].squeeze().cpu().numpy()
            
            ax.plot(alphas.cpu().numpy(), tau, "b-", label="τ (timing)")
            ax.plot(alphas.cpu().numpy(), sig, "r--", label="σ (width)")
            ax.set_xlabel("α")
            ax.set_ylabel("Parameter value")
            ax.set_title(f"{wave} wave parameters")
            ax.legend()
            
            if wave == target_component:
                ax.set_facecolor("#e8f5e9")  # Light green for target
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        
        return fig


def run_intervention_evaluation(
    model,
    dataloader,
    n_samples: int = 10,
    device: str = "cpu",
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Convenience function to run full intervention evaluation.
    
    Args:
        model: Trained EP-Prior model
        dataloader: DataLoader for getting baseline samples
        n_samples: Number of samples to average over
        device: Device for computation
        save_dir: Optional directory to save visualizations
        
    Returns:
        Dict with leakage metrics
    """
    tester = InterventionTester(model, device)
    results = tester.evaluate_all_interventions(dataloader, n_samples)
    
    print("\n" + "=" * 50)
    print("INTERVENTION LEAKAGE SUMMARY")
    print("=" * 50)
    
    for target, leakage in results.items():
        print(f"\nTarget: {target}")
        print(f"  Mean leakage: {leakage['mean_leakage']:.4f}")
        print(f"  Max leakage: {leakage['max_leakage']:.4f}")
    
    # Success criteria: <10% leakage
    all_mean = np.mean([r["mean_leakage"] for r in results.values()])
    print(f"\nOverall mean leakage: {all_mean:.4f}")
    print(f"Success (<10% leakage): {'✓' if all_mean < 0.1 else '✗'}")
    
    return results

