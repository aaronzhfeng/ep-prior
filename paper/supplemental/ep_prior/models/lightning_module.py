"""
EP-Prior Lightning Module: Self-Supervised Training Loop

Combines:
- StructuredEncoder: ECG → (z_P, z_QRS, z_T, z_HRV)
- GaussianWaveDecoder: (z_P, z_QRS, z_T) → Reconstructed ECG
- EP Constraint Losses: Soft penalties for physiological validity
- Optional: Contrastive loss for representation learning

Training Objective:
L = L_recon + λ_ep * L_ep + λ_contrast * L_contrast

Where:
- L_recon: MSE reconstruction loss
- L_ep: EP constraint violations (ordering, refractory, duration)
- L_contrast: Optional NT-Xent contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, Any
import math

from .structured_encoder import StructuredEncoder
from .gaussian_wave_decoder import GaussianWaveDecoder
from ..losses.ep_constraints import ep_constraint_loss, get_default_sigma_bounds


class EPPriorSSL(pl.LightningModule):
    """
    EP-Prior Self-Supervised Learning Module.
    
    Training modes:
    1. Reconstruction + EP constraints (default, simplest)
    2. + Contrastive learning (optional, requires augmented views)
    
    Args:
        input_channels: Number of ECG leads
        backbone_name: Encoder backbone variant
        d_P, d_QRS, d_T, d_HRV: Latent dimensions
        n_leads_decoder: Decoder output leads (can differ from input for single-lead experiments)
        lr: Learning rate
        weight_decay: AdamW weight decay
        lam_recon: Reconstruction loss weight
        lam_ep: EP constraint loss weight
        lam_contrast: Contrastive loss weight (0 to disable)
        ep_warmup_epochs: Epochs before EP constraints reach full weight
        pr_min, qt_min: Minimum intervals for EP constraints
        use_gates: Whether to gate EP constraints by wave presence
    """
    
    def __init__(
        self,
        # Architecture
        input_channels: int = 12,
        backbone_name: str = "xresnet1d50",
        d_P: int = 32,
        d_QRS: int = 128,
        d_T: int = 64,
        d_HRV: int = 32,
        n_leads_decoder: Optional[int] = None,
        # Optimizer
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Loss weights
        lam_recon: float = 1.0,
        lam_ep: float = 0.5,
        lam_contrast: float = 0.0,
        # EP constraint config
        ep_warmup_epochs: int = 10,
        pr_min: float = 0.04,
        qt_min: float = 0.08,
        use_gates: bool = True,
        sigma_bounds: Optional[Dict] = None,
        # Contrastive config
        temperature: float = 0.2,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default decoder leads to input channels
        if n_leads_decoder is None:
            n_leads_decoder = input_channels
        
        # Encoder
        self.encoder = StructuredEncoder(
            input_channels=input_channels,
            backbone_name=backbone_name,
            d_P=d_P,
            d_QRS=d_QRS,
            d_T=d_T,
            d_HRV=d_HRV,
        )
        
        # Decoder
        self.decoder = GaussianWaveDecoder(
            d_P=d_P,
            d_QRS=d_QRS,
            d_T=d_T,
            n_leads=n_leads_decoder,
        )
        
        # Optional: Projection head for contrastive learning
        if lam_contrast > 0:
            d_total = d_P + d_QRS + d_T + d_HRV
            self.projection_head = nn.Sequential(
                nn.Linear(d_total, d_total),
                nn.ReLU(),
                nn.Linear(d_total, projection_dim),
            )
        else:
            self.projection_head = None
        
        # Sigma bounds for duration constraints
        self.sigma_bounds = sigma_bounds or get_default_sigma_bounds()
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, Dict]:
        """
        Full forward pass: encode + decode.
        
        Args:
            x: ECG signal (B, C, T)
            
        Returns:
            z: Structured latents
            attn: Attention weights
            x_hat: Reconstructed signal
            params: Decoder parameters
        """
        z, attn = self.encoder(x, return_attention=True)
        
        # Only decode P, QRS, T (not HRV)
        z_decode = {"P": z["P"], "QRS": z["QRS"], "T": z["T"]}
        x_hat, params = self.decoder(z_decode, T=x.shape[-1])
        
        return z, attn, x_hat, params
    
    def compute_recon_loss(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor,
        robust: bool = False,
    ) -> torch.Tensor:
        """
        Reconstruction loss.
        
        Args:
            x: Original signal (B, C, T)
            x_hat: Reconstructed signal (B, C, T)
            robust: If True, use Huber loss instead of MSE
            
        Returns:
            loss: Scalar reconstruction loss
        """
        if robust:
            return F.smooth_l1_loss(x_hat, x)
        return F.mse_loss(x_hat, x)
    
    def compute_contrastive_loss(
        self,
        z1: Dict[str, torch.Tensor],
        z2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        NT-Xent contrastive loss between two views.
        
        Args:
            z1, z2: Latents from augmented views
            
        Returns:
            loss: Scalar contrastive loss
        """
        if self.projection_head is None:
            return torch.tensor(0.0, device=z1["P"].device)
        
        # Concatenate and project
        u1 = self.encoder.get_latent_concat(z1)
        u2 = self.encoder.get_latent_concat(z2)
        
        p1 = self.projection_head(u1)
        p2 = self.projection_head(u2)
        
        # L2 normalize
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        
        # NT-Xent loss
        return self._nt_xent_loss(p1, p2, self.hparams.temperature)
    
    def _nt_xent_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        temperature: float
    ) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        
        Positive pairs: (z1[i], z2[i])
        Negative pairs: all other combinations
        """
        B = z1.shape[0]
        device = z1.device
        
        # Concatenate: [z1; z2] of shape (2B, D)
        z = torch.cat([z1, z2], dim=0)
        
        # Cosine similarity matrix (2B, 2B)
        sim = torch.mm(z, z.t()) / temperature
        
        # Mask out self-similarity (diagonal) with large negative instead of -inf
        # Using -inf causes nan when multiplied by 0 in pos_mask
        mask = torch.eye(2 * B, device=device).bool()
        sim.masked_fill_(mask, -1e9)
        
        # Compute log_softmax
        log_softmax = F.log_softmax(sim, dim=1)
        
        # Extract positive pair log-probabilities using indexing (avoids 0 * -inf = nan)
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_i = torch.arange(B, device=device)
        loss = -(log_softmax[pos_i, pos_i + B].sum() + log_softmax[pos_i + B, pos_i].sum()) / (2 * B)
        
        return loss
    
    def get_ep_weight(self) -> float:
        """Get current EP constraint weight based on warmup schedule."""
        if self.hparams.ep_warmup_epochs <= 0:
            return self.hparams.lam_ep
        
        progress = min(1.0, self.current_epoch / self.hparams.ep_warmup_epochs)
        return self.hparams.lam_ep * progress
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Expected batch format:
        - batch["x"]: ECG signal (B, C, T)
        - batch["x_view1"], batch["x_view2"]: Augmented views (if contrastive)
        """
        x = batch["x"]
        
        # Forward pass
        z, attn, x_hat, params = self.forward(x)
        
        # Reconstruction loss
        l_recon = self.compute_recon_loss(x, x_hat)
        
        # EP constraint loss
        ep_losses = ep_constraint_loss(
            params,
            pr_min=self.hparams.pr_min,
            qt_min=self.hparams.qt_min,
            sigma_bounds=self.sigma_bounds,
            use_gates=self.hparams.use_gates,
        )
        l_ep = ep_losses["ep_total"]
        
        # Contrastive loss (if enabled)
        l_contrast = torch.tensor(0.0, device=x.device)
        if self.hparams.lam_contrast > 0 and "x_view1" in batch:
            z1, _ = self.encoder(batch["x_view1"])
            z2, _ = self.encoder(batch["x_view2"])
            l_contrast = self.compute_contrastive_loss(z1, z2)
        
        # Combine losses
        ep_weight = self.get_ep_weight()
        loss = (
            self.hparams.lam_recon * l_recon +
            ep_weight * l_ep +
            self.hparams.lam_contrast * l_contrast
        )
        
        # Logging
        self.log_dict({
            "train/loss": loss,
            "train/recon": l_recon,
            "train/ep_total": l_ep,
            "train/ep_order": ep_losses["ep_order"],
            "train/ep_pr": ep_losses["ep_pr"],
            "train/ep_qt": ep_losses["ep_qt"],
            "train/ep_sigma": ep_losses["ep_sigma"],
            "train/contrast": l_contrast,
            "train/ep_weight": ep_weight,
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log decoder parameters for monitoring
        if batch_idx % 100 == 0:
            self._log_decoder_params(params)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch["x"]
        
        z, attn, x_hat, params = self.forward(x)
        
        l_recon = self.compute_recon_loss(x, x_hat)
        ep_losses = ep_constraint_loss(
            params,
            pr_min=self.hparams.pr_min,
            qt_min=self.hparams.qt_min,
            sigma_bounds=self.sigma_bounds,
            use_gates=self.hparams.use_gates,
        )
        
        loss = self.hparams.lam_recon * l_recon + self.hparams.lam_ep * ep_losses["ep_total"]
        
        self.log_dict({
            "val/loss": loss,
            "val/recon": l_recon,
            "val/ep_total": ep_losses["ep_total"],
        }, prog_bar=True)
        
        return loss
    
    def _log_decoder_params(self, params: Dict):
        """Log decoder parameter statistics for monitoring."""
        for wave in ["P", "QRS", "T"]:
            tau = params[wave]["tau"]
            sig = params[wave]["sig"]
            gate = params[wave]["gate"]
            
            self.log_dict({
                f"params/{wave}_tau_mean": tau.mean(),
                f"params/{wave}_tau_std": tau.std(),
                f"params/{wave}_sig_mean": sig.mean(),
                f"params/{wave}_gate_mean": gate.mean(),
            })
    
    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Optional: cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Will be overridden by trainer max_epochs
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def get_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for downstream tasks.
        
        Args:
            x: ECG signal (B, C, T)
            
        Returns:
            Dict with structured latents and concatenated version
        """
        self.eval()
        with torch.no_grad():
            z, _ = self.encoder(x, return_attention=False)
            z["concat"] = self.encoder.get_latent_concat(z)
        return z


# =============================================================================
# Unit tests
# =============================================================================

def _test_lightning_module():
    """Test the EPPriorSSL module."""
    print("Testing EPPriorSSL Lightning Module...")
    
    # Create model
    model = EPPriorSSL(
        input_channels=12,
        backbone_name="xresnet1d50",
        d_P=32,
        d_QRS=128,
        d_T=64,
        d_HRV=32,
        lam_contrast=0.0,  # Disable contrastive for basic test
    )
    
    B = 4
    T = 1000
    
    # Create dummy batch
    batch = {"x": torch.randn(B, 12, T)}
    
    # Test forward
    z, attn, x_hat, params = model.forward(batch["x"])
    
    assert x_hat.shape == (B, 12, T), f"Expected (4, 12, 1000), got {x_hat.shape}"
    assert z["P"].shape == (B, 32)
    assert z["QRS"].shape == (B, 128)
    
    # Test training step
    loss = model.training_step(batch, batch_idx=0)
    assert loss.requires_grad
    
    print(f"  x_hat shape: {x_hat.shape}")
    print(f"  Training loss: {loss.item():.4f}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with contrastive
    print("\nTesting with contrastive loss...")
    model_contrast = EPPriorSSL(
        input_channels=12,
        backbone_name="xresnet1d50",
        lam_contrast=0.1,
    )
    
    batch_contrast = {
        "x": torch.randn(B, 12, T),
        "x_view1": torch.randn(B, 12, T),
        "x_view2": torch.randn(B, 12, T),
    }
    
    loss_contrast = model_contrast.training_step(batch_contrast, batch_idx=0)
    print(f"  Contrastive training loss: {loss_contrast.item():.4f}")
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(batch["x"])
    assert "concat" in embeddings
    assert embeddings["concat"].shape == (B, 32 + 128 + 64 + 32)
    
    print("✓ All Lightning module tests passed!")


if __name__ == "__main__":
    _test_lightning_module()

