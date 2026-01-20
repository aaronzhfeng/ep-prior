"""
StructuredEncoder: Attention-Pooled Encoder for EP-Prior

Takes ECG signals and produces structured latent representations:
- z_P: P-wave latent (atrial activity)
- z_QRS: QRS complex latent (ventricular depolarization)  
- z_T: T-wave latent (ventricular repolarization)
- z_HRV: Heart rate variability latent (RR dynamics)

Each wave component uses attention pooling over the temporal feature map,
allowing the model to learn which time regions correspond to each wave.

The backbone is xresnet1d50 from ecg-selfsupervised, modified to return
feature maps instead of classification logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
import os

# Add ecg-selfsupervised to path for imports
ECG_SSL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ecg-selfsupervised")
if ECG_SSL_PATH not in sys.path:
    sys.path.insert(0, ECG_SSL_PATH)


class BackboneFeatureExtractor(nn.Module):
    """
    Wraps xresnet1d to return feature maps instead of classification logits.
    
    The original xresnet1d is a Sequential:
    [stem..., maxpool, blocks..., head]
    
    We keep everything except the head to get the feature map.
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        backbone_name: str = "xresnet1d50",
    ):
        super().__init__()
        
        # Import here to avoid import errors if ecg-selfsupervised not available
        from clinical_ts.xresnet1d import xresnet1d50, xresnet1d34, xresnet1d18
        
        backbone_fn = {
            "xresnet1d50": xresnet1d50,
            "xresnet1d34": xresnet1d34,
            "xresnet1d18": xresnet1d18,
        }[backbone_name]
        
        # Create full backbone (will strip head)
        full_backbone = backbone_fn(input_channels=input_channels, num_classes=1)
        
        # The backbone is a Sequential. We want everything except the last element (head)
        # Structure: stem(3) + maxpool(1) + blocks(4) + head(1) = 9 elements for xresnet50
        # head is the last element
        self.features = nn.Sequential(*list(full_backbone.children())[:-1])
        
        # Determine output feature dimension by running a dummy forward
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 1000)
            dummy_output = self.features(dummy_input)
            self.feat_dim = dummy_output.shape[1]  # Channel dimension
            self._dummy_temporal_dim = dummy_output.shape[2]  # For reference
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ECG signal (B, C, T) where C=input_channels, T=time steps
            
        Returns:
            features: Feature map (B, D, L) where D=feat_dim, L=temporal length
        """
        return self.features(x)


class AttentionPooling(nn.Module):
    """
    Attention pooling over the temporal dimension.
    
    Learns to focus on specific time regions of the feature map.
    Returns both the pooled features and the attention weights (for visualization).
    """
    
    def __init__(self, feat_dim: int, temperature: float = 1.0):
        super().__init__()
        
        # Attention logits: 1x1 conv to scalar per time step
        self.attn_conv = nn.Conv1d(feat_dim, 1, kernel_size=1)
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map (B, D, L)
            
        Returns:
            pooled: Attention-pooled features (B, D)
            weights: Attention weights (B, L)
        """
        # Compute attention logits
        logits = self.attn_conv(x).squeeze(1)  # (B, L)
        
        # Softmax over temporal dimension
        weights = F.softmax(logits / self.temperature, dim=-1)  # (B, L)
        
        # Weighted sum over time
        # (B, D, L) * (B, 1, L) -> sum over L -> (B, D)
        pooled = torch.einsum("bdl,bl->bd", x, weights)
        
        return pooled, weights


class StructuredEncoder(nn.Module):
    """
    Encoder that produces structured latent representations.
    
    Architecture:
    1. Backbone (xresnet1d50) produces temporal feature map
    2. Separate attention pooling heads for P/QRS/T waves
    3. Global pooling for HRV (rhythm-level features)
    4. Linear projections to latent dimensions
    
    Args:
        input_channels: Number of ECG leads (default: 12)
        backbone_name: Which xresnet variant to use
        d_P: P-wave latent dimension
        d_QRS: QRS latent dimension  
        d_T: T-wave latent dimension
        d_HRV: HRV latent dimension
        dropout: Dropout rate for projections
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        backbone_name: str = "xresnet1d50",
        d_P: int = 32,
        d_QRS: int = 128,
        d_T: int = 64,
        d_HRV: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_P = d_P
        self.d_QRS = d_QRS
        self.d_T = d_T
        self.d_HRV = d_HRV
        self.d_total = d_P + d_QRS + d_T + d_HRV
        
        # Backbone feature extractor
        self.backbone = BackboneFeatureExtractor(
            input_channels=input_channels,
            backbone_name=backbone_name,
        )
        feat_dim = self.backbone.feat_dim
        
        # Wave-specific attention pooling
        self.attn_P = AttentionPooling(feat_dim)
        self.attn_QRS = AttentionPooling(feat_dim)
        self.attn_T = AttentionPooling(feat_dim)
        
        # Latent projections
        self.proj_P = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, d_P),
        )
        
        self.proj_QRS = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, d_QRS),
        )
        
        self.proj_T = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, d_T),
        )
        
        # HRV uses global pooling (rhythm-level, not wave-specific)
        self.proj_HRV = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, d_HRV),
        )
        
        # Optional: attention initialization to encourage temporal separation
        self._init_attention_biases()
    
    def _init_attention_biases(self):
        """
        Initialize attention biases to encourage P-early, QRS-middle, T-late.
        
        This is a soft prior that can be overridden by learning.
        """
        # We can't easily bias based on position, but we can initialize
        # the attention weights to be more uniform initially
        for attn in [self.attn_P, self.attn_QRS, self.attn_T]:
            nn.init.zeros_(attn.attn_conv.weight)
            nn.init.zeros_(attn.attn_conv.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True,
        return_features: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Encode ECG signal to structured latents.
        
        Args:
            x: ECG signal (B, C, T) where C=leads, T=time
            return_attention: Whether to return attention weights
            return_features: Whether to return backbone features
            
        Returns:
            z: Dict with keys "P", "QRS", "T", "HRV" containing latent vectors
            attn: Dict with attention weights (if return_attention=True)
            features: Backbone feature map (if return_features=True)
        """
        # Get backbone features
        features = self.backbone(x)  # (B, D, L)
        
        # Wave-specific attention pooling
        h_P, a_P = self.attn_P(features)
        h_QRS, a_QRS = self.attn_QRS(features)
        h_T, a_T = self.attn_T(features)
        
        # Global pooling for HRV
        h_HRV = features.mean(dim=-1)  # (B, D)
        
        # Project to latent dimensions
        z_P = self.proj_P(h_P)
        z_QRS = self.proj_QRS(h_QRS)
        z_T = self.proj_T(h_T)
        z_HRV = self.proj_HRV(h_HRV)
        
        z = {
            "P": z_P,
            "QRS": z_QRS,
            "T": z_T,
            "HRV": z_HRV,
        }
        
        attn = None
        if return_attention:
            attn = {
                "P": a_P,
                "QRS": a_QRS,
                "T": a_T,
            }
        
        if return_features:
            return z, attn, features
        
        return z, attn
    
    def get_latent_concat(self, z: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate all latents into single vector for downstream tasks."""
        return torch.cat([z["P"], z["QRS"], z["T"], z["HRV"]], dim=-1)
    
    def get_attention_entropy(self, attn: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute entropy of attention distributions.
        
        High entropy = diffuse attention (looking everywhere)
        Low entropy = focused attention (looking at specific region)
        
        Useful for monitoring whether attention is learning to focus.
        """
        entropies = {}
        for name, weights in attn.items():
            # weights: (B, L), sum to 1 over L
            # Entropy: -sum(p * log(p))
            eps = 1e-8
            entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # (B,)
            entropies[name] = entropy.mean()
        return entropies


class SingleLeadStructuredEncoder(StructuredEncoder):
    """
    Simplified encoder for single-lead experiments (MVP).
    """
    
    def __init__(
        self,
        backbone_name: str = "xresnet1d18",  # Smaller for single lead
        d_P: int = 32,
        d_QRS: int = 128,
        d_T: int = 64,
        d_HRV: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_channels=1,
            backbone_name=backbone_name,
            d_P=d_P,
            d_QRS=d_QRS,
            d_T=d_T,
            d_HRV=d_HRV,
            dropout=dropout,
        )


# =============================================================================
# Unit tests
# =============================================================================

def _test_encoder():
    """Test the StructuredEncoder."""
    print("Testing StructuredEncoder...")
    
    # Test 12-lead encoder
    encoder = StructuredEncoder(
        input_channels=12,
        backbone_name="xresnet1d50",
        d_P=32,
        d_QRS=128,
        d_T=64,
        d_HRV=32,
    )
    
    B = 4
    T = 1000  # 10 seconds at 100Hz
    x = torch.randn(B, 12, T)
    
    z, attn, features = encoder(x, return_attention=True, return_features=True)
    
    # Check latent shapes
    assert z["P"].shape == (B, 32), f"Expected (4, 32), got {z['P'].shape}"
    assert z["QRS"].shape == (B, 128), f"Expected (4, 128), got {z['QRS'].shape}"
    assert z["T"].shape == (B, 64), f"Expected (4, 64), got {z['T'].shape}"
    assert z["HRV"].shape == (B, 32), f"Expected (4, 32), got {z['HRV'].shape}"
    
    # Check attention shapes
    L = features.shape[2]  # Temporal length after backbone
    assert attn["P"].shape == (B, L), f"Expected (4, {L}), got {attn['P'].shape}"
    assert attn["QRS"].shape == (B, L)
    assert attn["T"].shape == (B, L)
    
    # Check attention sums to 1
    assert torch.allclose(attn["P"].sum(dim=-1), torch.ones(B), atol=1e-5)
    
    # Check concatenation
    z_concat = encoder.get_latent_concat(z)
    assert z_concat.shape == (B, 32 + 128 + 64 + 32)
    
    # Check gradients flow
    loss = sum(v.sum() for v in z.values())
    loss.backward()
    assert encoder.proj_P[0].weight.grad is not None
    
    # Check attention entropy
    entropies = encoder.get_attention_entropy(attn)
    print(f"  Feature map shape: {features.shape}")
    print(f"  z_P shape: {z['P'].shape}")
    print(f"  z_QRS shape: {z['QRS'].shape}")
    print(f"  z_T shape: {z['T'].shape}")
    print(f"  z_HRV shape: {z['HRV'].shape}")
    print(f"  Attention entropy (P): {entropies['P']:.3f}")
    print(f"  Attention entropy (QRS): {entropies['QRS']:.3f}")
    print(f"  Attention entropy (T): {entropies['T']:.3f}")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("✓ All encoder tests passed!")


def _test_single_lead():
    """Test single-lead encoder."""
    print("\nTesting SingleLeadStructuredEncoder...")
    
    encoder = SingleLeadStructuredEncoder()
    
    B = 4
    x = torch.randn(B, 1, 1000)
    
    z, attn = encoder(x, return_attention=True)
    
    assert z["P"].shape == (B, 32)
    assert z["QRS"].shape == (B, 128)
    
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("✓ Single-lead encoder tests passed!")


if __name__ == "__main__":
    _test_encoder()
    _test_single_lead()

