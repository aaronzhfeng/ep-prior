"""
Linear Probes for Downstream Evaluation

Trains linear classifiers on frozen embeddings for:
- Full concatenated latent → all labels
- Component-specific latents → corresponding pathologies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class LinearProbe(nn.Module):
    """Simple linear classifier for probe evaluation."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cpu",
) -> Tuple[LinearProbe, Dict[str, float]]:
    """
    Train a linear probe on embeddings.
    
    Args:
        train_embeddings: (N_train, D) training embeddings
        train_labels: (N_train, C) multi-label targets
        val_embeddings: (N_val, D) validation embeddings
        val_labels: (N_val, C) validation targets
        num_classes: Number of output classes
        lr: Learning rate
        epochs: Max training epochs
        batch_size: Batch size
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: Device to train on
        
    Returns:
        probe: Trained linear probe
        metrics: Dict with AUROC, AUPRC
    """
    input_dim = train_embeddings.shape[1]
    
    # Create model
    probe = LinearProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_embeddings = val_embeddings.to(device)
    val_labels = val_labels.to(device)
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        probe.train()
        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(batch_emb)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_embeddings)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best state
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    # Compute final metrics
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_embeddings)
        val_probs = torch.sigmoid(val_logits)
    
    metrics = compute_classification_metrics(
        val_labels.cpu().numpy(),
        val_probs.cpu().numpy(),
    )
    
    return probe, metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics for multi-label classification.
    
    Args:
        y_true: (N, C) ground truth labels
        y_prob: (N, C) predicted probabilities
        
    Returns:
        Dict with macro/micro AUROC, AUPRC
    """
    metrics = {}
    
    # Per-class metrics
    n_classes = y_true.shape[1]
    aurocs = []
    auprcs = []
    
    for i in range(n_classes):
        # Skip if no positive samples
        if y_true[:, i].sum() == 0:
            continue
        
        try:
            auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
            auprc = average_precision_score(y_true[:, i], y_prob[:, i])
            aurocs.append(auroc)
            auprcs.append(auprc)
        except ValueError:
            continue
    
    if aurocs:
        metrics["auroc_macro"] = np.mean(aurocs)
        metrics["auprc_macro"] = np.mean(auprcs)
        metrics["auroc_per_class"] = aurocs
        metrics["auprc_per_class"] = auprcs
    else:
        metrics["auroc_macro"] = 0.0
        metrics["auprc_macro"] = 0.0
    
    return metrics


def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract embeddings from a trained model (EP-Prior or Baseline).
    
    Args:
        model: Trained EPPriorSSL or BaselineSSL model
        dataloader: DataLoader with ECG data
        device: Device to use
        
    Returns:
        embeddings: Dict with 'concat' key (and P/QRS/T/HRV for EP-Prior)
        labels: Stacked labels if available
    """
    model.eval()
    model.to(device)
    
    # Check if model is EP-Prior (has structured encoder) or Baseline
    is_structured = hasattr(model.encoder, 'get_latent_concat')
    
    if is_structured:
        all_z = {"P": [], "QRS": [], "T": [], "HRV": [], "concat": []}
    else:
        all_z = {"concat": []}
    
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            x = batch["x"].to(device)
            
            if is_structured:
                # EP-Prior: encoder returns (z_dict, attn)
                z, _ = model.encoder(x, return_attention=False)
                all_z["P"].append(z["P"].cpu())
                all_z["QRS"].append(z["QRS"].cpu())
                all_z["T"].append(z["T"].cpu())
                all_z["HRV"].append(z["HRV"].cpu())
                all_z["concat"].append(model.encoder.get_latent_concat(z).cpu())
            else:
                # Baseline: encoder returns single tensor
                z = model.encoder(x)
                all_z["concat"].append(z.cpu())
            
            if "superclass" in batch:
                all_labels.append(batch["superclass"])
    
    # Concatenate
    embeddings = {k: torch.cat(v, dim=0) for k, v in all_z.items()}
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    
    return embeddings, labels

