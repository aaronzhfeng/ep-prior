"""
GaussianWaveDecoder: EP-Constrained Decoder for ECG Reconstruction

Reconstructs ECG signals from structured latent representations (z_P, z_QRS, z_T)
using a Gaussian wave basis model with physiologically-constrained parameters.

Model: x̂_t = Σ_{w ∈ {P, QRS, T}} A_w · exp(-(t - τ_w)² / 2σ_w²)

Parameters decoded from latents:
- τ_w: Wave timing (normalized to [0,1])
- σ_w: Wave width (with minimum bound for stability)
- A_w: Wave amplitude (per-lead, allowing sign changes)
- g_w: Gate/presence (for handling missing waves, e.g., absent P in AFib)

The QRS complex uses a mixture of K=3 Gaussians to capture Q/R/S morphology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GaussianWaveDecoder(nn.Module):
    """
    Decodes structured latents (z_P, z_QRS, z_T) into ECG waveforms.
    
    Key features:
    - Shared timing (τ) and width (σ) across leads (physiology: same event)
    - Per-lead amplitudes (physiology: different projections)
    - QRS mixture (K=3) for Q/R/S components
    - Presence gates for graceful handling of pathological cases
    
    Args:
        d_P: Dimension of P-wave latent
        d_QRS: Dimension of QRS latent  
        d_T: Dimension of T-wave latent
        n_leads: Number of ECG leads (default: 12)
        K: Tuple of mixture components per wave (P, QRS, T)
        sigma_min: Minimum width to prevent collapse
        hidden_dim: Hidden dimension for parameter MLPs
    """
    
    def __init__(
        self,
        d_P: int = 32,
        d_QRS: int = 128,
        d_T: int = 64,
        n_leads: int = 12,
        K: Tuple[int, int, int] = (1, 3, 1),
        sigma_min: float = 0.005,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.d_P = d_P
        self.d_QRS = d_QRS
        self.d_T = d_T
        self.n_leads = n_leads
        self.K_P, self.K_QRS, self.K_T = K
        self.sigma_min = sigma_min
        
        # Parameter prediction MLPs
        def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        
        # === P-wave parameters ===
        self.amp_P = make_mlp(d_P, n_leads * self.K_P)  # Per-lead, per-component
        self.tau_P = make_mlp(d_P, 1)                    # Shared timing
        self.sig_P = make_mlp(d_P, 1)                    # Shared width
        self.gate_P = make_mlp(d_P, 1)                   # Presence gate
        
        # === QRS parameters ===
        self.amp_QRS = make_mlp(d_QRS, n_leads * self.K_QRS)
        self.tau_QRS = make_mlp(d_QRS, 1)
        self.sig_QRS = make_mlp(d_QRS, 1)
        self.gate_QRS = make_mlp(d_QRS, 1)
        
        # QRS mixture offsets (Q before R, S after R)
        # Small fixed offsets in normalized time
        self.register_buffer(
            "qrs_offsets", 
            torch.tensor([-0.015, 0.0, 0.015]).view(1, 1, self.K_QRS)
        )
        
        # === T-wave parameters ===
        self.amp_T = make_mlp(d_T, n_leads * self.K_T)
        self.tau_T = make_mlp(d_T, 1)
        self.sig_T = make_mlp(d_T, 1)
        self.gate_T = make_mlp(d_T, 1)
        
        # Learnable output scale to match normalized ECG amplitude
        # Initialize high (~15) since MLP amplitudes start small
        self.output_scale = nn.Parameter(torch.tensor(15.0))
        
        # Initialize biases for reasonable defaults
        self._init_biases()
    
    def _init_biases(self):
        """Initialize biases for physiologically reasonable defaults."""
        # Timing biases: P early, QRS middle, T late (in normalized [0,1])
        # sigmoid(0) = 0.5, so we shift
        nn.init.constant_(self.tau_P[-1].bias, -1.5)   # ~0.18
        nn.init.constant_(self.tau_QRS[-1].bias, -0.5) # ~0.38
        nn.init.constant_(self.tau_T[-1].bias, 0.5)    # ~0.62
        
        # Width biases: reasonable starting widths
        nn.init.constant_(self.sig_P[-1].bias, -2.0)   # softplus(-2) + min ≈ 0.13
        nn.init.constant_(self.sig_QRS[-1].bias, -2.5) # narrower
        nn.init.constant_(self.sig_T[-1].bias, -1.5)   # wider
        
        # Gate biases: start with all waves present
        nn.init.constant_(self.gate_P[-1].bias, 2.0)   # sigmoid(2) ≈ 0.88
        nn.init.constant_(self.gate_QRS[-1].bias, 3.0) # sigmoid(3) ≈ 0.95
        nn.init.constant_(self.gate_T[-1].bias, 2.0)
    
    def _decode_wave(
        self,
        z: torch.Tensor,
        amp_head: nn.Module,
        tau_head: nn.Module,
        sig_head: nn.Module,
        gate_head: nn.Module,
        K: int,
        t: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode a single wave component.
        
        Args:
            z: Latent vector (B, d)
            amp_head, tau_head, sig_head, gate_head: Parameter MLPs
            K: Number of mixture components
            t: Time grid (1, 1, T) normalized to [0, 1]
            offsets: Optional mixture component offsets (1, 1, K)
            
        Returns:
            wave: Reconstructed wave (B, n_leads, T)
            params: Dict of decoded parameters
        """
        B = z.shape[0]
        device = z.device
        
        # Decode parameters
        A = amp_head(z).view(B, self.n_leads, K)  # (B, L, K)
        tau = torch.sigmoid(tau_head(z))          # (B, 1) in [0, 1]
        sig = F.softplus(sig_head(z)) + self.sigma_min  # (B, 1) > sigma_min
        # Gate with minimum value to prevent collapse to trivial solution
        gate_min = 0.1
        gate = gate_min + (1.0 - gate_min) * torch.sigmoid(gate_head(z))  # (B, 1) in [0.1, 1]
        
        # Expand timing for mixture components
        # tau_k shape: (B, 1, K, 1) for broadcasting with time
        if offsets is not None:
            tau_k = tau.view(B, 1, 1, 1) + offsets.to(device).unsqueeze(-1)  # (B, 1, K, 1)
        else:
            tau_k = tau.view(B, 1, 1, 1).expand(B, 1, K, 1)
        
        # Width: (B, 1, 1, 1)
        sig_e = sig.view(B, 1, 1, 1)
        
        # Time grid: (1, 1, 1, T)
        t_expanded = t.view(1, 1, 1, -1)
        
        # Gaussian basis: (B, 1, K, T)
        # exp(-0.5 * ((t - tau) / sigma)^2)
        basis = torch.exp(-0.5 * ((t_expanded - tau_k) / sig_e).pow(2))
        
        # Apply amplitudes: (B, L, K, 1) * (B, 1, K, T) -> sum over K -> (B, L, T)
        A_expanded = A.view(B, self.n_leads, K, 1)
        wave = (A_expanded * basis).sum(dim=2)  # (B, L, T)
        
        # Apply presence gate
        wave = gate.view(B, 1, 1) * wave
        
        params = {
            "A": A,
            "tau": tau,
            "sig": sig,
            "gate": gate,
        }
        
        return wave, params
    
    def forward(
        self, 
        z: Dict[str, torch.Tensor], 
        T: int,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Decode structured latents to ECG waveform.
        
        Args:
            z: Dict with keys "P", "QRS", "T" containing latent vectors
            T: Number of time steps to generate
            return_components: If True, also return individual wave components
            
        Returns:
            x_hat: Reconstructed ECG (B, n_leads, T)
            params: Dict of parameters for each wave
            (optional) components: Dict of individual wave reconstructions
        """
        B = z["P"].shape[0]
        device = z["P"].device
        
        # Normalized time grid [0, 1]
        t = torch.linspace(0.0, 1.0, T, device=device)
        
        # Decode each wave
        x_P, params_P = self._decode_wave(
            z["P"], self.amp_P, self.tau_P, self.sig_P, self.gate_P,
            self.K_P, t, offsets=None
        )
        
        x_QRS, params_QRS = self._decode_wave(
            z["QRS"], self.amp_QRS, self.tau_QRS, self.sig_QRS, self.gate_QRS,
            self.K_QRS, t, offsets=self.qrs_offsets
        )
        
        x_T, params_T = self._decode_wave(
            z["T"], self.amp_T, self.tau_T, self.sig_T, self.gate_T,
            self.K_T, t, offsets=None
        )
        
        # Sum waves and apply learnable scale
        x_hat = (x_P + x_QRS + x_T) * self.output_scale  # (B, n_leads, T)
        
        params = {
            "P": params_P,
            "QRS": params_QRS,
            "T": params_T,
        }
        
        if return_components:
            # Scale components too for consistency
            components = {
                "P": x_P * self.output_scale, 
                "QRS": x_QRS * self.output_scale, 
                "T": x_T * self.output_scale
            }
            return x_hat, params, components
        
        return x_hat, params
    
    def get_timing_order(self, params: Dict) -> torch.Tensor:
        """
        Get wave timings for constraint checking.
        
        Returns:
            timings: (B, 3) tensor of [tau_P, tau_QRS, tau_T]
        """
        tau_P = params["P"]["tau"]      # (B, 1)
        tau_QRS = params["QRS"]["tau"]  # (B, 1)
        tau_T = params["T"]["tau"]      # (B, 1)
        
        return torch.cat([tau_P, tau_QRS, tau_T], dim=1)  # (B, 3)


class SingleLeadGaussianWaveDecoder(GaussianWaveDecoder):
    """
    Simplified decoder for single-lead experiments (MVP).
    
    Same as GaussianWaveDecoder but with n_leads=1 by default.
    Useful for initial development and debugging.
    """
    
    def __init__(
        self,
        d_P: int = 32,
        d_QRS: int = 128,
        d_T: int = 64,
        K: Tuple[int, int, int] = (1, 3, 1),
        sigma_min: float = 0.005,
        hidden_dim: int = 128,  # Smaller for single lead
    ):
        super().__init__(
            d_P=d_P,
            d_QRS=d_QRS,
            d_T=d_T,
            n_leads=1,
            K=K,
            sigma_min=sigma_min,
            hidden_dim=hidden_dim,
        )


# =============================================================================
# Unit tests
# =============================================================================

def _test_decoder():
    """Quick sanity check for the decoder."""
    print("Testing GaussianWaveDecoder...")
    
    # Create decoder
    decoder = GaussianWaveDecoder(d_P=32, d_QRS=128, d_T=64, n_leads=12)
    
    # Create dummy latents
    B = 4
    z = {
        "P": torch.randn(B, 32),
        "QRS": torch.randn(B, 128),
        "T": torch.randn(B, 64),
    }
    
    # Forward pass
    x_hat, params, components = decoder(z, T=1000, return_components=True)
    
    # Check shapes
    assert x_hat.shape == (B, 12, 1000), f"Expected (4, 12, 1000), got {x_hat.shape}"
    assert components["P"].shape == (B, 12, 1000)
    assert components["QRS"].shape == (B, 12, 1000)
    assert components["T"].shape == (B, 12, 1000)
    
    # Check parameter shapes
    assert params["P"]["tau"].shape == (B, 1)
    assert params["P"]["sig"].shape == (B, 1)
    assert params["P"]["gate"].shape == (B, 1)
    assert params["P"]["A"].shape == (B, 12, 1)  # K_P = 1
    assert params["QRS"]["A"].shape == (B, 12, 3)  # K_QRS = 3
    
    # Check timing order helper
    timings = decoder.get_timing_order(params)
    assert timings.shape == (B, 3)
    
    # Check gradients flow
    loss = x_hat.sum()
    loss.backward()
    assert decoder.tau_P[0].weight.grad is not None
    
    # Check sigma stays positive
    assert (params["P"]["sig"] > 0).all()
    assert (params["QRS"]["sig"] > 0).all()
    assert (params["T"]["sig"] > 0).all()
    
    # Check gates in [0, 1]
    assert (params["P"]["gate"] >= 0).all() and (params["P"]["gate"] <= 1).all()
    
    print(f"  x_hat shape: {x_hat.shape}")
    print(f"  tau_P range: [{params['P']['tau'].min():.3f}, {params['P']['tau'].max():.3f}]")
    print(f"  tau_QRS range: [{params['QRS']['tau'].min():.3f}, {params['QRS']['tau'].max():.3f}]")
    print(f"  tau_T range: [{params['T']['tau'].min():.3f}, {params['T']['tau'].max():.3f}]")
    print(f"  Parameter count: {sum(p.numel() for p in decoder.parameters()):,}")
    print("✓ All tests passed!")


if __name__ == "__main__":
    _test_decoder()

