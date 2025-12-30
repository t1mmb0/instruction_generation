import numpy as np
import torch
import pandas as pd


class FeatureImportanceAnalyzer:
    """
    Gradient×Input Feature Importance for PyG link prediction models
    (encode(x, edge_index) + decode(z, edge_label_index)).

    Returns a per-feature importance score aggregated over the loader.
    """

    def __init__(self, model, data_loader, feature_names, device, criterion):
        self.model = model
        self.data_loader = data_loader
        self.feature_names = list(feature_names)
        self.device = device
        self.criterion = criterion

        if len(self.feature_names) == 0:
            raise ValueError("feature_names must not be empty.")

    @torch.inference_mode(False)  # ensure gradients are allowed even if outer context uses inference_mode
    def compute_importance(self, normalize=True, use_grad_times_input=True, eps=1e-12):
        """
        Args:
            normalize: if True, divide by total number of nodes seen.
            use_grad_times_input: if True use |grad * x| else use |grad|.
            eps: numerical stability.
        Returns:
            pd.DataFrame with columns: feature, importance, importance_norm
        """
        self.model.eval()

        F = len(self.feature_names)
        importance = np.zeros(F, dtype=np.float64)
        total_nodes = 0

        for batch in self.data_loader:
            batch = batch.to(self.device)

            # Create a fresh leaf tensor for x so we can read x.grad safely
            x = batch.x.detach().clone().requires_grad_(True)
            batch.x = x  # make model use our grad-enabled x

            # Forward pass (same as your eval step)
            z = self.model.encode(batch.x, batch.edge_index)
            logits = self.model.decode(z, batch.edge_label_index)
            y = batch.edge_label.float()

            loss = self.criterion(logits, y)

            # Backward to get gradients w.r.t. x
            self.model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()

            grad_x = x.grad  # shape [num_nodes, num_features]
            if grad_x is None:
                raise RuntimeError(
                    "x.grad is None. Check that you did not run under no_grad/inference_mode "
                    "and that x.requires_grad_(True) is set."
                )

            # Aggregate per feature
            if use_grad_times_input:
                contrib = (grad_x * x).abs()  # Gradient×Input
            else:
                contrib = grad_x.abs()

            # Sum over nodes -> [F]
            contrib_f = contrib.sum(dim=0).detach().cpu().numpy()

            # Defensive check: feature dimension matches names
            if contrib_f.shape[0] != F:
                raise ValueError(
                    f"Feature dim mismatch: got {contrib_f.shape[0]} features in batch.x "
                    f"but {F} feature_names were provided."
                )

            importance += contrib_f
            total_nodes += batch.x.size(0)

        if normalize and total_nodes > 0:
            importance = importance / float(total_nodes)

        # Also provide a normalized-to-sum-1 column for easy "dead feature" spotting
        s = float(importance.sum())
        importance_norm = importance / (s + eps)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "importance_norm": importance_norm,
        }).sort_values("importance", ascending=False, ignore_index=True)

        return df

    @staticmethod
    def find_dead_features(df, col="importance_norm", threshold=1e-4):
        """
        Simple helper to flag 'dead' features.
        threshold: features below this normalized importance are considered unused.
        """
        dead = df[df[col] < threshold].copy()
        return dead
