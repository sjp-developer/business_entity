"""
Supervised Pairwise Matching Model Module.
Implements LightGBM matching model, hard-negative training, and F0.5 threshold optimization.
"""
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Set, Tuple

from src.config import RANDOM_SEED, N_JOBS
from src.features import FEATURE_NAMES
from src.evaluation import evaluate_predictions

class MatchingModel:
    """
    LightGBM pairwise matching classifier for Entity Resolution.
    """
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6, num_leaves: int = 31):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        self.best_threshold = 0.5
        
    def train(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None):
        """Train LightGBM model on pair features X and binary labels y."""
        print(f"[Model] Training LightGBM on {len(X)} candidate pairs (Positives: {np.sum(y == 1)}, Negatives: {np.sum(y == 0)})...", flush=True)
        self.model.fit(X, y, sample_weight=sample_weights)
        print("[Model] Training complete.", flush=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability P(match) for input features X."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, gt_map: Dict[str, Set[str]], candidate_scores: Dict[str, List[Tuple[str, float]]], thresholds: List[float] = None) -> float:
        """
        Finds the decision threshold that maximizes Macro F0.5 on validation set.
        
        candidate_scores: {s1_id: [(cand_id, score), ...]}
        """
        if thresholds is None:
            thresholds = [round(x, 2) for x in np.arange(0.20, 0.85, 0.02)]
            
        print("[Model] Optimizing decision threshold against Macro F0.5...", flush=True)
        best_f05 = -1.0
        best_thresh = 0.5
        
        for t in thresholds:
            pred_map = {}
            for s1_id, scored_cands in candidate_scores.items():
                selected = {cand_id for cand_id, score in scored_cands if score >= t}
                pred_map[s1_id] = selected
                
            metrics = evaluate_predictions(gt_map, pred_map)
            macro_f05 = metrics['macro_f05']
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thresh = t
                
            print(f"  Threshold {t:.2f} -> Macro F0.5: {macro_f05:.4f} (Prec: {metrics['macro_precision']:.4f}, Rec: {metrics['macro_recall']:.4f}, Singletons: {metrics['singleton_accuracy']:.2f}%)")
            
        print(f"[Model] Optimal threshold: {best_thresh:.2f} with Macro F0.5 = {best_f05:.4f}", flush=True)
        self.best_threshold = best_thresh
        return best_thresh

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns dict of feature name -> feature importance score."""
        importances = self.model.feature_importances_
        return dict(zip(FEATURE_NAMES, importances))
