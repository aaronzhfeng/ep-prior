"""
Concept Predictability Evaluation

Tests whether structured latents predict corresponding pathologies:
- z_QRS → Bundle Branch Blocks (LBBB, RBBB, wide QRS)
- z_P → Atrial abnormalities (AFib, AFL, P-wave abnormal)
- z_T → QT/ST abnormalities (prolonged QT, T-wave abnormal)

This validates that the latent structure has physiological meaning.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# PTB-XL SCP code groupings for concept validation
CONCEPT_GROUPS = {
    "QRS_related": {
        "description": "Conduction abnormalities (should be predicted by z_QRS)",
        "codes": [
            "CLBBB", "CRBBB",  # Complete BBB
            "ILBBB", "IRBBB",  # Incomplete BBB
            "IVCD",            # Intraventricular conduction delay
            "WPW",             # Wolff-Parkinson-White
            "LAFB", "LPFB",    # Fascicular blocks
        ],
        "superclass_proxy": "CD",  # Conduction Disturbance
    },
    "P_related": {
        "description": "Atrial abnormalities (should be predicted by z_P)",
        "codes": [
            "AFIB", "AFLT",    # Atrial fibrillation/flutter
            "SARRH", "SBRAD",  # Sinus arrhythmia/bradycardia
            "PAC",             # Premature atrial contractions
            "RAO", "LAO",      # Right/Left atrial overload
        ],
        "superclass_proxy": None,  # Mixed across classes
    },
    "T_related": {
        "description": "Repolarization abnormalities (should be predicted by z_T)",
        "codes": [
            "NDT", "NST_",     # Nonspecific T/ST changes
            "LNGQT",           # Long QT
            "INVT",            # T-wave inversion
            "LOWT",            # Low T-wave amplitude
            "TAB_",            # T-wave abnormality
        ],
        "superclass_proxy": "STTC",  # ST/T change
    },
}


class ConceptPredictabilityEvaluator:
    """
    Evaluates whether structured latents predict clinically meaningful concepts.
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        device: str = "cpu",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        
        # Extract embeddings
        self._extract_embeddings()
    
    def _extract_embeddings(self):
        """Extract embeddings and labels from datasets."""
        self.model.eval()
        self.model.to(self.device)
        
        self.train_embeddings = self._extract_from_dataset(self.train_dataset)
        self.test_embeddings = self._extract_from_dataset(self.test_dataset)
    
    def _extract_from_dataset(
        self,
        dataset,
    ) -> Dict:
        """Extract embeddings and superclass labels."""
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        
        all_z = {"P": [], "QRS": [], "T": [], "HRV": [], "concat": []}
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                z, _ = self.model.encoder(x, return_attention=False)
                
                all_z["P"].append(z["P"].cpu())
                all_z["QRS"].append(z["QRS"].cpu())
                all_z["T"].append(z["T"].cpu())
                all_z["HRV"].append(z["HRV"].cpu())
                all_z["concat"].append(self.model.encoder.get_latent_concat(z).cpu())
                
                if "superclass" in batch:
                    all_labels.append(batch["superclass"])
        
        embeddings = {k: torch.cat(v, dim=0).numpy() for k, v in all_z.items()}
        labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None
        
        return {"embeddings": embeddings, "labels": labels}
    
    def evaluate_superclass_prediction(self) -> pd.DataFrame:
        """
        Test: Which embedding best predicts which superclass?
        
        PTB-XL superclasses: NORM, MI, STTC, CD, HYP
        
        Hypothesis:
        - z_QRS should excel at CD (conduction disturbance)
        - z_T should excel at STTC (ST/T changes)
        """
        superclass_names = ["NORM", "MI", "STTC", "CD", "HYP"]
        embedding_keys = ["P", "QRS", "T", "HRV", "concat"]
        
        results = []
        
        for emb_key in embedding_keys:
            X_train = self.train_embeddings["embeddings"][emb_key]
            X_test = self.test_embeddings["embeddings"][emb_key]
            
            for class_idx, class_name in enumerate(superclass_names):
                y_train = self.train_embeddings["labels"][:, class_idx]
                y_test = self.test_embeddings["labels"][:, class_idx]
                
                # Skip if no positive samples
                if y_train.sum() < 10 or y_test.sum() < 5:
                    continue
                
                # Train logistic regression
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_prob = clf.predict_proba(X_test)[:, 1]
                
                try:
                    auroc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    auroc = 0.5
                
                results.append({
                    "embedding": emb_key,
                    "superclass": class_name,
                    "auroc": auroc,
                    "n_train_pos": int(y_train.sum()),
                    "n_test_pos": int(y_test.sum()),
                })
        
        return pd.DataFrame(results)
    
    def get_expected_associations(self) -> Dict[str, List[str]]:
        """
        Return expected latent-to-pathology associations.
        
        These are the hypotheses we want to validate.
        """
        return {
            "QRS": ["CD"],       # QRS latent should predict Conduction Disturbance
            "T": ["STTC"],       # T latent should predict ST/T changes
            "P": [],             # P-wave abnormalities not well represented in superclasses
            "concat": ["NORM", "MI", "STTC", "CD", "HYP"],  # Full latent predicts all
        }
    
    def compute_selectivity_score(
        self,
        results_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute selectivity: how much better is each latent at its expected target?
        
        Selectivity = AUROC(target) - mean(AUROC(non-targets))
        """
        expected = self.get_expected_associations()
        selectivity = {}
        
        for emb_key in ["P", "QRS", "T"]:
            emb_results = results_df[results_df["embedding"] == emb_key]
            
            if len(emb_results) == 0:
                continue
            
            targets = expected.get(emb_key, [])
            
            if not targets:
                selectivity[emb_key] = 0.0
                continue
            
            target_aurocs = emb_results[
                emb_results["superclass"].isin(targets)
            ]["auroc"].mean()
            
            non_target_aurocs = emb_results[
                ~emb_results["superclass"].isin(targets)
            ]["auroc"].mean()
            
            selectivity[emb_key] = target_aurocs - non_target_aurocs
        
        return selectivity


def run_concept_evaluation(
    model,
    train_dataset,
    test_dataset,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full concept predictability evaluation.
    
    Args:
        model: Trained EP-Prior model
        train_dataset: Training dataset with labels
        test_dataset: Test dataset with labels
        device: Device for computation
        save_path: Optional path to save results
        
    Returns:
        results_df: Full results table
        summary: Summary metrics
    """
    evaluator = ConceptPredictabilityEvaluator(
        model, train_dataset, test_dataset, device
    )
    
    results_df = evaluator.evaluate_superclass_prediction()
    selectivity = evaluator.compute_selectivity_score(results_df)
    
    print("\n" + "=" * 60)
    print("CONCEPT PREDICTABILITY RESULTS")
    print("=" * 60)
    
    # Pivot table for easy reading
    pivot = results_df.pivot(
        index="superclass",
        columns="embedding",
        values="auroc",
    ).round(3)
    
    print("\nAUROC by (superclass, embedding):")
    print(pivot)
    
    print("\nSelectivity scores:")
    for emb, score in selectivity.items():
        print(f"  {emb}: {score:.4f}")
    
    # Check key hypotheses
    print("\n--- Key Hypothesis Validation ---")
    
    # z_QRS → CD
    qrs_cd = results_df[
        (results_df["embedding"] == "QRS") & 
        (results_df["superclass"] == "CD")
    ]["auroc"].values
    
    if len(qrs_cd) > 0:
        print(f"z_QRS predicts CD: AUROC = {qrs_cd[0]:.3f}")
        print(f"  Target: >0.7, Actual: {'✓' if qrs_cd[0] > 0.7 else '✗'}")
    
    # z_T → STTC
    t_sttc = results_df[
        (results_df["embedding"] == "T") & 
        (results_df["superclass"] == "STTC")
    ]["auroc"].values
    
    if len(t_sttc) > 0:
        print(f"z_T predicts STTC: AUROC = {t_sttc[0]:.3f}")
        print(f"  Target: >0.7, Actual: {'✓' if t_sttc[0] > 0.7 else '✗'}")
    
    if save_path:
        results_df.to_csv(save_path, index=False)
    
    summary = {
        "selectivity": selectivity,
        "pivot": pivot,
    }
    
    return results_df, summary

