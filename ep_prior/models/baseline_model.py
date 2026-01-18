"""
Capacity-Matched Baseline Model

Same encoder architecture and parameter count as EP-Prior, but:
- Unstructured latent space (single vector instead of z_P, z_QRS, z_T, z_HRV)
- Generic MLP decoder (no EP constraints)
- No physiological structure

This provides a fair comparison to isolate the effect of EP structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple

from .structured_encoder import BackboneFeatureExtractor


class GenericEncoder(nn.Module):
    """
    Generic encoder that outputs a single unstructured latent vector.
    Same backbone and total latent dimension as StructuredEncoder.
    """
    
    def __init__(
        self,
        backbone_name: str = "xresnet1d50",
        input_channels: int = 12,
        d_latent: int = 256,  # Same total as d_P + d_QRS + d_T + d_HRV
    ):
        super().__init__()
        
        self.backbone = BackboneFeatureExtractor(input_channels, backbone_name)
        self.feat_dim = self.backbone.feat_dim
        
        # Global pooling + projection (same capacity as structured version)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, d_latent),
        )
        
        self.d_latent = d_latent
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) ECG signal
            
        Returns:
            z: (B, d_latent) latent vector
        """
        feat = self.backbone(x)  # (B, D, L)
        pooled = feat.mean(dim=-1)  # (B, D)
        z = self.proj(pooled)  # (B, d_latent)
        return z


class GenericDecoder(nn.Module):
    """
    Generic MLP decoder for reconstruction.
    No physiological structure or constraints.
    """
    
    def __init__(
        self,
        d_latent: int = 256,
        n_leads: int = 12,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.n_leads = n_leads
        
        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output projection (will be reshaped)
        self.output_proj = nn.Linear(hidden_dim, n_leads * 100)  # 100 features per timestep chunk
        
        # Upsample to full length
        self.upsample = nn.ConvTranspose1d(n_leads, n_leads, kernel_size=16, stride=10, padding=3)
    
    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            z: (B, d_latent) latent vector
            T: target sequence length
            
        Returns:
            x_hat: (B, n_leads, T) reconstructed signal
        """
        B = z.shape[0]
        
        h = self.decoder(z)  # (B, hidden_dim)
        out = self.output_proj(h)  # (B, n_leads * 100)
        out = out.view(B, self.n_leads, 100)  # (B, n_leads, 100)
        
        # Upsample to target length
        x_hat = self.upsample(out)  # (B, n_leads, ~1000)
        
        # Interpolate to exact length
        x_hat = F.interpolate(x_hat, size=T, mode='linear', align_corners=False)
        
        return x_hat


class BaselineSSL(pl.LightningModule):
    """
    Capacity-matched baseline for fair comparison with EP-Prior.
    
    Same:
    - Encoder backbone (xresnet1d50)
    - Total latent dimension (256)
    - Training objective (reconstruction + optional contrastive)
    
    Different:
    - Unstructured latent (single vector)
    - Generic MLP decoder
    - No EP constraints
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        backbone_name: str = "xresnet1d50",
        d_latent: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lam_recon: float = 1.0,
        lam_contrast: float = 0.0,
        temperature: float = 0.2,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        self.encoder = GenericEncoder(
            backbone_name=backbone_name,
            input_channels=input_channels,
            d_latent=d_latent,
        )
        
        # Decoder
        self.decoder = GenericDecoder(
            d_latent=d_latent,
            n_leads=input_channels,
        )
        
        # Optional projection head for contrastive
        if lam_contrast > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(d_latent, d_latent),
                nn.ReLU(),
                nn.Linear(d_latent, projection_dim),
            )
        else:
            self.projection_head = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        x_hat = self.decoder(z, T=x.shape[-1])
        return z, x_hat
    
    def compute_recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        return F.mse_loss(x_hat, x)
    
    def compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent contrastive loss."""
        if self.projection_head is None:
            return torch.tensor(0.0, device=z1.device)
        
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        
        return self._nt_xent_loss(p1, p2, self.hparams.temperature)
    
    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """NT-Xent loss (same implementation as EP-Prior)."""
        B = z1.shape[0]
        device = z1.device
        
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / temperature
        
        mask = torch.eye(2 * B, device=device).bool()
        sim.masked_fill_(mask, -1e9)
        
        log_softmax = F.log_softmax(sim, dim=1)
        
        pos_i = torch.arange(B, device=device)
        loss = -(log_softmax[pos_i, pos_i + B].sum() + log_softmax[pos_i + B, pos_i].sum()) / (2 * B)
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["x"]
        
        z, x_hat = self.forward(x)
        l_recon = self.compute_recon_loss(x, x_hat)
        
        # Contrastive loss
        l_contrast = torch.tensor(0.0, device=x.device)
        if self.hparams.lam_contrast > 0 and "x_view1" in batch:
            z1 = self.encoder(batch["x_view1"])
            z2 = self.encoder(batch["x_view2"])
            l_contrast = self.compute_contrastive_loss(z1, z2)
        
        loss = self.hparams.lam_recon * l_recon + self.hparams.lam_contrast * l_contrast
        
        self.log_dict({
            "train/loss": loss,
            "train/recon": l_recon,
            "train/contrast": l_contrast,
        }, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch["x"]
        z, x_hat = self.forward(x)
        l_recon = self.compute_recon_loss(x, x_hat)
        
        self.log_dict({
            "val/loss": l_recon,
            "val/recon": l_recon,
        }, prog_bar=True)
        
        return l_recon
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
    
    def get_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract embeddings for downstream evaluation."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return {"concat": z}


if __name__ == "__main__":
    # Test
    model = BaselineSSL()
    x = torch.randn(4, 12, 1000)
    z, x_hat = model(x)
    print(f"Input: {x.shape}")
    print(f"Latent: {z.shape}")
    print(f"Output: {x_hat.shape}")
    print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

