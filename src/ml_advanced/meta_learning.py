"""Meta-learning for rapid market-regime adaptation.

Implements MAML-style (Model-Agnostic Meta-Learning) and prototypical
approaches so the trading models can quickly adapt to new market regimes
with only a handful of recent examples.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MAMLAdapter:
    """Model-Agnostic Meta-Learning (MAML) adapter for PyTorch models.

    Given a base model that was meta-trained across multiple market
    regimes, this class provides fast fine-tuning on a small support set
    from a new (or changed) regime.
    """

    def __init__(
        self,
        base_model: Any,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        outer_lr: float = 1e-3,
        first_order: bool = True,
    ) -> None:
        """Initialise the MAML adapter.

        Args:
            base_model: A PyTorch ``nn.Module`` to be meta-trained.
            inner_lr: Learning rate for task-specific gradient steps.
            inner_steps: Number of inner-loop gradient updates.
            outer_lr: Meta-optimizer learning rate.
            first_order: When *True* use the first-order MAML
                approximation (FOMAML) for efficiency.
        """
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.first_order = first_order
        self._meta_optimizer: Any = None

    def _clone_model(self) -> Any:
        """Return a deep copy of the base model."""
        return deepcopy(self.base_model)

    def adapt(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
        loss_fn: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """Fast-adapt the model to a support set from a new market regime.

        Args:
            support_X: Support features of shape ``(k, n_features)``.
            support_y: Support targets of shape ``(k,)`` or ``(k, horizon)``.
            loss_fn: Loss function accepting ``(predictions, targets)``.
                Defaults to MSE.

        Returns:
            A fine-tuned copy of the model (original is unchanged).
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch is required for MAML adaptation.")
            raise

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        adapted = self._clone_model()
        X_t = torch.FloatTensor(support_X)
        y_t = torch.FloatTensor(support_y)

        for step in range(self.inner_steps):
            preds = adapted(X_t)
            loss = loss_fn(preds, y_t)
            grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=not self.first_order)
            with torch.no_grad():
                for param, grad in zip(adapted.parameters(), grads):
                    param -= self.inner_lr * grad
            logger.debug("MAML inner step %d/%d – loss: %.6f", step + 1, self.inner_steps, loss.item())

        return adapted

    def meta_train(
        self,
        task_generator: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        n_tasks: int = 100,
        meta_batch_size: int = 8,
        loss_fn: Optional[Callable[..., Any]] = None,
    ) -> List[float]:
        """Run the meta-training outer loop.

        Args:
            task_generator: Callable that returns
                ``(support_X, support_y, query_X, query_y)`` for one task.
            n_tasks: Total number of meta-training tasks.
            meta_batch_size: Tasks per meta-gradient update.
            loss_fn: Loss function; defaults to MSE.

        Returns:
            List of meta-losses per outer step.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch is required for meta-training.")
            raise

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self._meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.outer_lr)
        meta_losses: List[float] = []

        for outer_step in range(n_tasks // meta_batch_size):
            self._meta_optimizer.zero_grad()
            outer_loss = torch.tensor(0.0)

            for _ in range(meta_batch_size):
                sup_X, sup_y, qry_X, qry_y = task_generator()
                adapted = self.adapt(sup_X, sup_y, loss_fn)
                q_X_t = torch.FloatTensor(qry_X)
                q_y_t = torch.FloatTensor(qry_y)
                q_preds = adapted(q_X_t)
                outer_loss = outer_loss + loss_fn(q_preds, q_y_t)

            outer_loss = outer_loss / meta_batch_size
            outer_loss.backward()
            self._meta_optimizer.step()
            meta_losses.append(outer_loss.item())
            logger.info("Meta outer step %d – meta_loss: %.6f", outer_step + 1, outer_loss.item())

        return meta_losses


class PrototypicalNetworkAdapter:
    """Prototypical Networks for few-shot market-regime classification.

    Learns an embedding space where each market regime is represented by
    the centroid (prototype) of its embedded support examples.
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialise the prototypical network.

        Args:
            embedding_dim: Dimension of the learned embedding space.
            hidden_dim: Hidden layer width of the encoder MLP.
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self._encoder: Any = None
        self._prototypes: Dict[int, Any] = {}

    def _build_encoder(self, input_dim: int) -> Any:
        """Construct the encoder MLP.

        Args:
            input_dim: Number of input features.
        """
        try:
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch required for encoder construction.")
            raise

        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def compute_prototypes(
        self,
        support_X: np.ndarray,
        support_labels: np.ndarray,
    ) -> None:
        """Compute class prototypes from the support set.

        Args:
            support_X: Support features ``(n_support, n_features)``.
            support_labels: Integer class labels ``(n_support,)``.
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch required.")
            raise

        if self._encoder is None:
            self._encoder = self._build_encoder(support_X.shape[1])

        self._encoder.eval()
        X_t = torch.FloatTensor(support_X)
        with torch.no_grad():
            embeddings = self._encoder(X_t)

        for cls in np.unique(support_labels):
            mask = support_labels == cls
            self._prototypes[int(cls)] = embeddings[mask].mean(dim=0)
        logger.info("Computed %d prototypes.", len(self._prototypes))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify query samples by nearest prototype.

        Args:
            X: Query features ``(n_samples, n_features)``.

        Returns:
            Predicted class labels ``(n_samples,)``.
        """
        if not self._prototypes:
            raise RuntimeError("Call compute_prototypes() first.")
        try:
            import torch
        except ImportError:
            logger.error("PyTorch required.")
            raise

        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            embeddings = self._encoder(X_t)

        classes = sorted(self._prototypes.keys())
        prototypes = [self._prototypes[c] for c in classes]
        # Euclidean distance to each prototype
        dists = torch.stack([torch.norm(embeddings - p, dim=1) for p in prototypes], dim=1)
        nearest = torch.argmin(dists, dim=1).numpy()
        return np.array([classes[i] for i in nearest])


class RegimeDetector:
    """Unsupervised market-regime detector using Gaussian Mixture Models.

    Detects hidden market states (bull, bear, sideways, volatile) from
    feature representations and can rapidly re-identify the current
    regime for adaptive strategy switching.
    """

    def __init__(self, n_regimes: int = 4, covariance_type: str = "full", random_state: int = 42) -> None:
        """Initialise the regime detector.

        Args:
            n_regimes: Number of latent market regimes.
            covariance_type: GMM covariance structure (``"full"``, ``"tied"``, etc.).
            random_state: Random seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._gmm: Any = None
        self._scaler: Any = None

    def fit(self, X: np.ndarray) -> "RegimeDetector":
        """Fit the GMM to feature data.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.

        Returns:
            ``self``.
        """
        try:
            from sklearn.mixture import GaussianMixture  # noqa: PLC0415
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        except ImportError:
            logger.error("scikit-learn is required for RegimeDetector.")
            raise

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._gmm.fit(X_scaled)
        logger.info("RegimeDetector fitted with %d regimes.", self.n_regimes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign regime labels to *X*.

        Args:
            X: Feature matrix.

        Returns:
            Integer regime labels of shape ``(n_samples,)``.
        """
        if self._gmm is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return self._gmm.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft regime probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix ``(n_samples, n_regimes)``.
        """
        if self._gmm is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return self._gmm.predict_proba(self._scaler.transform(X))
