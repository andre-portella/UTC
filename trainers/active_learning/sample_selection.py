import sys
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

class Selector:
    """Selects query samples for Active Learning based on uncertainty or representativeness strategies."""

    def __init__(self, task_phi, device, U_index):
        """Initializes the Selector with model representations, device, and unlabeled pool indices."""

        self.task_phi = task_phi
        self.device = device
        self.U_index = U_index

    def select(self, strategy, n_query, features, probs):
        """Dispatches the query selection to the specified acquisition strategy."""

        strategies = {
            "centroid": self._centroid_selection,
            "entropy": self._entropy_selection,
            "confidence": self._confidence_selection,
            "margin": self._margin_sampling_selection,
            "margin_confidence": self._margin_sampling_confidence_selection,
        }

        if strategy not in strategies:
            raise ValueError(f"Strategy '{strategy}' not valid.")

        return strategies[strategy](n_query, features, probs)

    # =========================================================
    # CENTROID
    # =========================================================
    def _centroid_selection(self, n_query, features, probs):
        """Selects representative samples closest to the geometric center of the feature space."""

        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        centroid = np.mean(features_np, axis=0).reshape(1, -1)
        dists = cdist(features_np, centroid, metric="euclidean").flatten()
        sorted_idx = np.argsort(dists)[:n_query]
        return sorted_idx.tolist(), None

    # =========================================================
    # ENTROPY
    # =========================================================
    def _entropy_selection(self, n_query, features, probs):
        """Selects samples with the lowest predictive entropy."""
        
        entropies = -(probs * torch.log(torch.clamp(probs, min=sys.float_info.epsilon))).sum(dim=1)
        sorted_idx = torch.argsort(entropies).cpu().numpy()[:n_query]
        return sorted_idx.tolist(), None

    # =========================================================
    # CONFIDENCE
    # =========================================================
    def _confidence_selection(self, n_query, features, probs):
        """Selects samples with the highest maximum predicted class probability."""

        confidence = probs.max(dim=1).values
        sorted_idx = torch.argsort(confidence, descending=True).cpu().numpy()[:n_query]
        return sorted_idx.tolist(), None

    # =========================================================
    # MARGIN
    # =========================================================
    def _margin_sampling_selection(self, n_query, features, probs):
        """Selects samples based on the difference between the two highest predicted probabilities."""

        top2 = probs.topk(2, dim=1).values
        margin_scores = top2[:, 0] - top2[:, 1]
        sorted_idx = torch.argsort(margin_scores, descending=True).cpu().numpy()[:n_query]
        return sorted_idx.tolist(), None

    # =========================================================
    # MARGIN + CONFIDENCE
    # =========================================================
    def _margin_sampling_confidence_selection(self, n_query, features, probs):
        """Selects samples by weighting margin difference with the highest class confidence."""

        top2 = probs.topk(2, dim=1).values
        margin_scores = top2[:, 0] - top2[:, 1]
        confidence = top2[:, 0]
        combined_score = margin_scores * confidence
        sorted_idx = torch.argsort(combined_score, descending=True).cpu().numpy()[:n_query]
        return sorted_idx.tolist(), None


    def run(self, strategy, n_query, features_cluster, probs_cluster):
        """Selects indices based on the specified strategy."""
        return self.select(strategy, n_query, features_cluster, probs_cluster)