"""
EP Constraint Losses: Soft Penalties for Electrophysiology Constraints

Implements soft penalty losses that encourage the decoder to produce
physiologically plausible wave parameters. These are SOFT constraints
(not hard masks) to allow the model to learn when violations are informative
(e.g., wide QRS in bundle branch block).

Constraints:
1. Ordering: τ_P < τ_QRS < τ_T (P-wave before QRS before T-wave)
2. Refractory: |τ_QRS - τ_P| > ΔPR_min (minimum PR interval)
3. Duration bounds: σ_w ∈ [σ_min, σ_max] (wave width limits)
4. QT interval: |τ_T - τ_QRS| > ΔQT_min (minimum QT interval)

All constraints are gated by wave presence to handle pathological cases.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def soft_hinge(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Soft hinge loss: smoothed version of ReLU.
    
    softplus(x) = (1/beta) * log(1 + exp(beta * x))
    
    Approaches ReLU as beta -> infinity, smoother for finite beta.
    Better gradients than ReLU near zero.
    """
    return F.softplus(x, beta=beta)


def ordering_loss(
    tau_P: torch.Tensor,
    tau_QRS: torch.Tensor, 
    tau_T: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Soft ordering constraint: τ_P < τ_QRS < τ_T
    
    Penalizes when waves are out of order.
    
    Args:
        tau_P, tau_QRS, tau_T: Wave timings (B, 1) in [0, 1]
        margin: Extra separation margin (default: 0)
        
    Returns:
        loss: Scalar ordering violation loss
    """
    # P should come before QRS
    loss_p_qrs = soft_hinge(tau_P - tau_QRS + margin)
    
    # QRS should come before T
    loss_qrs_t = soft_hinge(tau_QRS - tau_T + margin)
    
    return (loss_p_qrs + loss_qrs_t).mean()


def refractory_loss(
    tau_P: torch.Tensor,
    tau_QRS: torch.Tensor,
    tau_T: torch.Tensor,
    pr_min: float = 0.06,
    qt_min: float = 0.12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Refractory period constraints: minimum intervals between waves.
    
    PR interval: Time from P-wave onset to QRS onset (normally 120-200ms)
    QT interval: Time from QRS onset to T-wave end (normally 350-450ms)
    
    In normalized time [0,1] for 10s ECG:
    - pr_min=0.06 → 600ms minimum (conservative)
    - qt_min=0.12 → 1.2s minimum (conservative)
    
    Note: These are intentionally loose to avoid over-constraining.
    The model should learn tighter constraints from data.
    
    Args:
        tau_P, tau_QRS, tau_T: Wave timings (B, 1)
        pr_min: Minimum PR interval in normalized time
        qt_min: Minimum QT interval in normalized time
        
    Returns:
        loss_pr: PR interval violation loss
        loss_qt: QT interval violation loss
    """
    # PR interval should be >= pr_min
    pr_interval = tau_QRS - tau_P
    loss_pr = soft_hinge(pr_min - pr_interval)
    
    # QT interval should be >= qt_min  
    qt_interval = tau_T - tau_QRS
    loss_qt = soft_hinge(qt_min - qt_interval)
    
    return loss_pr.mean(), loss_qt.mean()


def duration_bounds_loss(
    sig: torch.Tensor,
    sig_min: float,
    sig_max: float,
) -> torch.Tensor:
    """
    Duration bounds constraint: σ ∈ [σ_min, σ_max]
    
    Penalizes wave widths outside physiological bounds.
    
    Args:
        sig: Wave width (B, 1)
        sig_min: Minimum allowed width
        sig_max: Maximum allowed width
        
    Returns:
        loss: Duration bounds violation loss
    """
    # Penalize too narrow
    loss_min = soft_hinge(sig_min - sig)
    
    # Penalize too wide
    loss_max = soft_hinge(sig - sig_max)
    
    return (loss_min + loss_max).mean()


def ep_constraint_loss(
    params: Dict[str, Dict[str, torch.Tensor]],
    pr_min: float = 0.04,
    qt_min: float = 0.08,
    order_margin: float = 0.0,
    sigma_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    use_gates: bool = True,
    gate_threshold: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Combined EP constraint loss.
    
    Aggregates all EP constraints with optional gating by wave presence.
    When use_gates=True, constraints are scaled by gate values so that
    constraints on absent waves (e.g., P-wave in AFib) are relaxed.
    
    Args:
        params: Decoder output parameters dict with structure:
            params["P"]["tau"], params["P"]["sig"], params["P"]["gate"]
            params["QRS"]["tau"], params["QRS"]["sig"], params["QRS"]["gate"]
            params["T"]["tau"], params["T"]["sig"], params["T"]["gate"]
        pr_min: Minimum PR interval (normalized time)
        qt_min: Minimum QT interval (normalized time)
        order_margin: Extra margin for ordering constraint
        sigma_bounds: Dict of (min, max) bounds per wave, e.g.:
            {"P": (0.005, 0.04), "QRS": (0.003, 0.02), "T": (0.006, 0.06)}
        use_gates: Whether to gate constraints by wave presence
        gate_threshold: Below this gate value, constraint is fully relaxed
        
    Returns:
        losses: Dict containing individual loss terms and total
    """
    # Extract timing parameters
    tau_P = params["P"]["tau"].squeeze(-1)      # (B,)
    tau_QRS = params["QRS"]["tau"].squeeze(-1)
    tau_T = params["T"]["tau"].squeeze(-1)
    
    # Extract gates
    g_P = params["P"]["gate"].squeeze(-1)       # (B,)
    g_QRS = params["QRS"]["gate"].squeeze(-1)
    g_T = params["T"]["gate"].squeeze(-1)
    
    # === Ordering loss ===
    l_order_raw = (
        soft_hinge(tau_P - tau_QRS + order_margin) +
        soft_hinge(tau_QRS - tau_T + order_margin)
    )
    
    # === Refractory losses ===
    l_pr_raw = soft_hinge(pr_min - (tau_QRS - tau_P))
    l_qt_raw = soft_hinge(qt_min - (tau_T - tau_QRS))
    
    # === Apply gating ===
    if use_gates:
        # Ordering requires all three waves
        gate_all = g_P * g_QRS * g_T
        l_order = l_order_raw * gate_all
        
        # PR requires P and QRS
        gate_pr = g_P * g_QRS
        l_pr = l_pr_raw * gate_pr
        
        # QT requires QRS and T
        gate_qt = g_QRS * g_T
        l_qt = l_qt_raw * gate_qt
    else:
        l_order = l_order_raw
        l_pr = l_pr_raw
        l_qt = l_qt_raw
    
    # === Duration bounds loss ===
    l_sigma = torch.tensor(0.0, device=tau_P.device)
    
    if sigma_bounds is not None:
        for wave_name in ["P", "QRS", "T"]:
            if wave_name in sigma_bounds:
                sig = params[wave_name]["sig"].squeeze(-1)
                s_min, s_max = sigma_bounds[wave_name]
                
                term = soft_hinge(s_min - sig) + soft_hinge(sig - s_max)
                
                if use_gates:
                    gate = params[wave_name]["gate"].squeeze(-1)
                    term = term * gate
                
                l_sigma = l_sigma + term.mean()
    
    # === Aggregate ===
    losses = {
        "ep_order": l_order.mean(),
        "ep_pr": l_pr.mean(),
        "ep_qt": l_qt.mean(),
        "ep_sigma": l_sigma,
    }
    losses["ep_total"] = (
        losses["ep_order"] + 
        losses["ep_pr"] + 
        losses["ep_qt"] + 
        losses["ep_sigma"]
    )
    
    return losses


def get_default_sigma_bounds(normalized: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Get default sigma bounds based on typical ECG wave durations.
    
    In a 10-second ECG at normalized time [0, 1]:
    - P-wave: 80-120ms → 0.008-0.012 (but Gaussian σ is ~1/3 of duration)
    - QRS: 80-120ms normally, up to 200ms pathological
    - T-wave: 160-200ms
    
    We use generous bounds to allow pathological cases.
    
    Args:
        normalized: If True, return bounds in normalized time [0,1] for 10s
                   If False, return bounds in seconds
                   
    Returns:
        sigma_bounds: Dict of (min, max) per wave
    """
    if normalized:
        # For 10-second ECG normalized to [0, 1]
        # These are Gaussian σ values, not full wave durations
        return {
            "P": (0.003, 0.025),    # ~30-250ms effective width
            "QRS": (0.002, 0.020),  # ~20-200ms (allows wide QRS)
            "T": (0.005, 0.040),    # ~50-400ms (T-wave is broader)
        }
    else:
        # In seconds (for reference)
        return {
            "P": (0.03, 0.25),
            "QRS": (0.02, 0.20),
            "T": (0.05, 0.40),
        }


class EPConstraintLoss(torch.nn.Module):
    """
    Module wrapper for EP constraint loss.
    
    Convenient for use in training loops with configurable hyperparameters.
    """
    
    def __init__(
        self,
        pr_min: float = 0.04,
        qt_min: float = 0.08,
        order_margin: float = 0.0,
        sigma_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        use_gates: bool = True,
        lambda_order: float = 1.0,
        lambda_pr: float = 1.0,
        lambda_qt: float = 0.5,
        lambda_sigma: float = 0.5,
    ):
        super().__init__()
        
        self.pr_min = pr_min
        self.qt_min = qt_min
        self.order_margin = order_margin
        self.sigma_bounds = sigma_bounds or get_default_sigma_bounds()
        self.use_gates = use_gates
        
        # Loss weights
        self.lambda_order = lambda_order
        self.lambda_pr = lambda_pr
        self.lambda_qt = lambda_qt
        self.lambda_sigma = lambda_sigma
    
    def forward(
        self, 
        params: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted EP constraint loss.
        
        Args:
            params: Decoder parameters
            
        Returns:
            total_loss: Weighted sum of constraint losses
            losses: Dict of individual (unweighted) loss terms
        """
        losses = ep_constraint_loss(
            params,
            pr_min=self.pr_min,
            qt_min=self.qt_min,
            order_margin=self.order_margin,
            sigma_bounds=self.sigma_bounds,
            use_gates=self.use_gates,
        )
        
        total = (
            self.lambda_order * losses["ep_order"] +
            self.lambda_pr * losses["ep_pr"] +
            self.lambda_qt * losses["ep_qt"] +
            self.lambda_sigma * losses["ep_sigma"]
        )
        
        return total, losses


# =============================================================================
# Unit tests
# =============================================================================

def _test_constraints():
    """Test EP constraint losses."""
    print("Testing EP constraint losses...")
    
    B = 8
    device = "cpu"
    
    # Create mock decoder params
    # Case 1: Correct ordering (should have low loss)
    params_good = {
        "P": {
            "tau": torch.tensor([[0.15]] * B),   # Early
            "sig": torch.tensor([[0.010]] * B),
            "gate": torch.tensor([[0.9]] * B),
        },
        "QRS": {
            "tau": torch.tensor([[0.35]] * B),   # Middle
            "sig": torch.tensor([[0.008]] * B),
            "gate": torch.tensor([[0.95]] * B),
        },
        "T": {
            "tau": torch.tensor([[0.60]] * B),   # Late
            "sig": torch.tensor([[0.020]] * B),
            "gate": torch.tensor([[0.9]] * B),
        },
    }
    
    # Case 2: Wrong ordering (should have high loss)
    params_bad = {
        "P": {
            "tau": torch.tensor([[0.50]] * B),   # Too late!
            "sig": torch.tensor([[0.010]] * B),
            "gate": torch.tensor([[0.9]] * B),
        },
        "QRS": {
            "tau": torch.tensor([[0.35]] * B),
            "sig": torch.tensor([[0.008]] * B),
            "gate": torch.tensor([[0.95]] * B),
        },
        "T": {
            "tau": torch.tensor([[0.30]] * B),   # Too early!
            "sig": torch.tensor([[0.020]] * B),
            "gate": torch.tensor([[0.9]] * B),
        },
    }
    
    # Case 3: AFib (absent P-wave, low gate)
    params_afib = {
        "P": {
            "tau": torch.tensor([[0.50]] * B),   # Wrong timing, but gated out
            "sig": torch.tensor([[0.010]] * B),
            "gate": torch.tensor([[0.1]] * B),   # Low gate = absent P
        },
        "QRS": {
            "tau": torch.tensor([[0.35]] * B),
            "sig": torch.tensor([[0.008]] * B),
            "gate": torch.tensor([[0.95]] * B),
        },
        "T": {
            "tau": torch.tensor([[0.60]] * B),
            "sig": torch.tensor([[0.020]] * B),
            "gate": torch.tensor([[0.9]] * B),
        },
    }
    
    # Test with default bounds
    sigma_bounds = get_default_sigma_bounds()
    
    losses_good = ep_constraint_loss(params_good, sigma_bounds=sigma_bounds)
    losses_bad = ep_constraint_loss(params_bad, sigma_bounds=sigma_bounds)
    losses_afib = ep_constraint_loss(params_afib, sigma_bounds=sigma_bounds, use_gates=True)
    
    print(f"  Good ordering total: {losses_good['ep_total']:.4f}")
    print(f"    - order: {losses_good['ep_order']:.4f}")
    print(f"    - PR: {losses_good['ep_pr']:.4f}")
    print(f"    - QT: {losses_good['ep_qt']:.4f}")
    print(f"    - sigma: {losses_good['ep_sigma']:.4f}")
    
    print(f"  Bad ordering total: {losses_bad['ep_total']:.4f}")
    print(f"    - order: {losses_bad['ep_order']:.4f}")
    
    print(f"  AFib (gated) total: {losses_afib['ep_total']:.4f}")
    print(f"    - order: {losses_afib['ep_order']:.4f}")
    
    # Verify expected behavior
    assert losses_good["ep_order"] < losses_bad["ep_order"], \
        "Good ordering should have lower loss than bad"
    assert losses_afib["ep_order"] < losses_bad["ep_order"], \
        "AFib (gated) should have lower loss than ungated bad ordering"
    
    # Test module wrapper
    loss_module = EPConstraintLoss()
    total, losses = loss_module(params_good)
    assert total.requires_grad == False  # No learnable params in loss
    
    print("✓ All constraint tests passed!")


if __name__ == "__main__":
    _test_constraints()

