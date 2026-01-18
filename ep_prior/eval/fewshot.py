"""
Few-Shot Evaluation for EP-Prior

Evaluates sample efficiency by training linear probes on 
{10, 50, 100, 500} examples per class and measuring downstream performance.

This tests the PAC-Bayes prediction: EP-Prior should show largest
gains over baselines in low-n regimes.
"""

import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from .probes import train_linear_probe, extract_embeddings, compute_classification_metrics


class FewShotEvaluator:
    """
    Evaluates few-shot performance across different shot sizes.
    
    For each shot size k, samples k examples per class (for multi-label,
    ensures at least k positives per label), trains a linear probe,
    and evaluates on the full test set.
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        shot_sizes: List[int] = [10, 50, 100, 500],
        num_seeds: int = 3,
        label_key: str = "superclass",
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained EP-Prior model
            train_dataset: Full training dataset
            test_dataset: Test dataset
            shot_sizes: List of k values for k-shot evaluation
            num_seeds: Number of random seeds for each k
            label_key: Key for labels in dataset
            device: Device for training
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.shot_sizes = shot_sizes
        self.num_seeds = num_seeds
        self.label_key = label_key
        self.device = device
        
        # Pre-extract test embeddings (shared across all shots)
        self._extract_test_embeddings()
    
    def _extract_test_embeddings(self):
        """Extract embeddings for test set once."""
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=64, 
            shuffle=False,
            num_workers=0,
        )
        self.test_embeddings, self.test_labels = extract_embeddings(
            self.model, test_loader, self.device
        )
    
    def _sample_fewshot_indices(
        self, 
        k: int, 
        seed: int,
    ) -> List[int]:
        """
        Sample k examples per class for few-shot training.
        
        For multi-label, uses greedy sampling to ensure coverage.
        """
        np.random.seed(seed)
        
        # Build label-to-indices mapping
        n_samples = len(self.train_dataset)
        label_to_indices = defaultdict(list)
        
        for idx in range(n_samples):
            sample = self.train_dataset[idx]
            if self.label_key in sample:
                labels = sample[self.label_key]
                for label_idx, val in enumerate(labels):
                    if val > 0:
                        label_to_indices[label_idx].append(idx)
        
        # Greedy sampling: ensure at least k per label
        selected = set()
        
        for label_idx in label_to_indices:
            indices = label_to_indices[label_idx]
            # Filter out already selected
            available = [i for i in indices if i not in selected]
            
            # Sample up to k
            n_sample = min(k, len(available))
            if n_sample > 0:
                sampled = np.random.choice(available, size=n_sample, replace=False)
                selected.update(sampled)
        
        return list(selected)
    
    def _evaluate_single(
        self,
        train_indices: List[int],
        embedding_key: str = "concat",
    ) -> Dict[str, float]:
        """Evaluate with a specific set of training indices."""
        # Create subset dataloader
        subset = Subset(self.train_dataset, train_indices)
        train_loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
        
        # Extract train embeddings
        train_embeddings, train_labels = extract_embeddings(
            self.model, train_loader, self.device
        )
        
        # Get specific embedding type
        train_emb = train_embeddings[embedding_key]
        test_emb = self.test_embeddings[embedding_key]
        
        # Train probe
        num_classes = train_labels.shape[1]
        probe, metrics = train_linear_probe(
            train_emb, train_labels,
            test_emb, self.test_labels,
            num_classes=num_classes,
            device=self.device,
        )
        
        return metrics
    
    def evaluate_all(
        self,
        embedding_keys: List[str] = ["concat", "P", "QRS", "T"],
    ) -> pd.DataFrame:
        """
        Run full few-shot evaluation.
        
        Returns DataFrame with columns:
        - shot_size, seed, embedding, auroc_macro, auprc_macro
        """
        results = []
        
        for k in tqdm(self.shot_sizes, desc="Shot sizes"):
            print(f"\nEvaluating {k}-shot...")
            
            for seed in tqdm(range(self.num_seeds), desc=f"Seeds ({k}-shot)", leave=False):
                # Sample indices
                indices = self._sample_fewshot_indices(k, seed)
                print(f"  Seed {seed}: {len(indices)} samples")
                
                for emb_key in embedding_keys:
                    metrics = self._evaluate_single(indices, emb_key)
                    
                    results.append({
                        "shot_size": k,
                        "seed": seed,
                        "embedding": emb_key,
                        "auroc_macro": metrics.get("auroc_macro", 0),
                        "auprc_macro": metrics.get("auprc_macro", 0),
                        "n_train": len(indices),
                    })
        
        return pd.DataFrame(results)
    
    def summarize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results across seeds.
        
        Returns DataFrame with mean ± std for each (shot_size, embedding).
        """
        summary = df.groupby(["shot_size", "embedding"]).agg({
            "auroc_macro": ["mean", "std"],
            "auprc_macro": ["mean", "std"],
            "n_train": "mean",
        }).round(4)
        
        return summary


def run_fewshot_evaluation(
    model,
    train_dataset,
    test_dataset,
    shot_sizes: List[int] = [10, 50, 100, 500],
    num_seeds: int = 3,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run full few-shot evaluation.
    
    Args:
        model: Trained EP-Prior model
        train_dataset: Training dataset with labels
        test_dataset: Test dataset with labels
        shot_sizes: List of k values
        num_seeds: Number of seeds per k
        device: Device for evaluation
        save_path: Optional path to save results CSV
        
    Returns:
        results_df: Raw results
        summary_df: Aggregated results
    """
    evaluator = FewShotEvaluator(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        shot_sizes=shot_sizes,
        num_seeds=num_seeds,
        device=device,
    )
    
    results_df = evaluator.evaluate_all()
    summary_df = evaluator.summarize_results(results_df)
    
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    
    return results_df, summary_df

