"""Model interpretability using SHAP and LIME.

All heavy ML-library imports (shap, lime) are deferred inside functions
so this module can be imported without those libraries installed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Wrapper around the ``shap`` library for model explanation.

    Supports tree-based models (TreeExplainer) and any callable model
    (KernelExplainer) automatically.
    """

    def __init__(self, model: Any, background_data: Optional[np.ndarray] = None) -> None:
        """Initialise the SHAP explainer.

        Args:
            model: Fitted model object (sklearn, XGBoost, LightGBM, etc.).
            background_data: Background dataset for KernelExplainer.
                Required when *model* is not tree-based.
        """
        self.model = model
        self.background_data = background_data
        self._explainer: Any = None

    def _build_explainer(self) -> None:
        """Instantiate the appropriate SHAP explainer for *model*."""
        try:
            import shap  # noqa: PLC0415
        except ImportError:
            logger.error("shap is not installed. Install it with: pip install shap")
            raise

        tree_types = self._tree_model_types()
        if isinstance(self.model, tree_types):
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("Using SHAP TreeExplainer.")
        else:
            if self.background_data is None:
                raise ValueError("background_data is required for KernelExplainer.")
            self._explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            logger.info("Using SHAP KernelExplainer.")

    @staticmethod
    def _tree_model_types() -> Tuple[Any, ...]:
        """Return a tuple of tree-model base classes available at runtime."""
        types: List[Any] = []
        try:
            import xgboost  # noqa: PLC0415

            types.append(xgboost.XGBModel)
        except ImportError:
            pass
        try:
            import lightgbm  # noqa: PLC0415

            types.append(lightgbm.LGBMModel)
        except ImportError:
            pass
        try:
            from sklearn.ensemble import (  # noqa: PLC0415
                RandomForestClassifier,
                RandomForestRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )

            types.extend([RandomForestClassifier, RandomForestRegressor,
                         GradientBoostingClassifier, GradientBoostingRegressor])
        except ImportError:
            pass
        try:
            from catboost import CatBoost  # noqa: PLC0415

            types.append(CatBoost)
        except ImportError:
            pass
        return tuple(types) if types else (object,)

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """Compute SHAP values for *X*.

        Args:
            X: Input array of shape ``(n_samples, n_features)``.
            feature_names: Optional list of feature names for logging.
            check_additivity: Passed to ``shap_values()`` for tree explainers.

        Returns:
            SHAP values array of shape ``(n_samples, n_features)``.
        """
        if self._explainer is None:
            self._build_explainer()

        try:
            shap_values = self._explainer.shap_values(X, check_additivity=check_additivity)
        except TypeError:
            shap_values = self._explainer.shap_values(X)

        # For multi-output tree models shap_values is a list; take class 1.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.array(shap_values)

    def global_importance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute mean absolute SHAP values as global feature importance.

        Args:
            X: Input array.
            feature_names: Feature names; defaults to ``f0, f1, …``.

        Returns:
            Dictionary mapping feature name to mean |SHAP| value, sorted
            descending.
        """
        shap_values = self.explain(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        names = feature_names or [f"f{i}" for i in range(mean_abs.shape[0])]
        importance = dict(zip(names, mean_abs.tolist()))
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    def summary_plot(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Render a SHAP summary plot.

        Args:
            X: Input array.
            feature_names: Optional feature names.
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError:
            logger.error("shap not installed.")
            raise
        shap_values = self.explain(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names)

    def waterfall_plot(self, X: np.ndarray, sample_idx: int = 0) -> None:
        """Render a SHAP waterfall plot for a single sample.

        Args:
            X: Input array.
            sample_idx: Row index to explain.
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError:
            logger.error("shap not installed.")
            raise
        if self._explainer is None:
            self._build_explainer()
        explanation = self._explainer(X[sample_idx: sample_idx + 1])
        shap.plots.waterfall(explanation[0])


class LIMEExplainer:
    """Wrapper around ``lime`` for local model explanations.

    Works with any model exposing a ``predict`` (regression) or
    ``predict_proba`` (classification) method.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        mode: str = "regression",
        discretize_continuous: bool = True,
    ) -> None:
        """Initialise the LIME explainer.

        Args:
            training_data: Representative training samples used to estimate
                feature distributions (shape ``(n_samples, n_features)``).
            feature_names: Names for each feature column.
            mode: ``"regression"`` or ``"classification"``.
            discretize_continuous: Whether to discretise continuous features.
        """
        self.training_data = training_data
        self.feature_names = feature_names or [f"f{i}" for i in range(training_data.shape[1])]
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self._explainer: Any = None

    def _build_explainer(self) -> None:
        """Instantiate the LIME tabular explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer  # noqa: PLC0415
        except ImportError:
            logger.error("lime is not installed. Install it with: pip install lime")
            raise

        self._explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            mode=self.mode,
            discretize_continuous=self.discretize_continuous,
        )
        logger.info("LIME tabular explainer created.")

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, float]:
        """Explain a single prediction with LIME.

        Args:
            instance: 1-D feature array for the sample to explain.
            predict_fn: Model prediction function accepting a 2-D array.
            num_features: Maximum features to include in the explanation.
            num_samples: Number of perturbed samples used for LIME.

        Returns:
            Dictionary mapping feature description to contribution weight.
        """
        if self._explainer is None:
            self._build_explainer()

        explanation = self._explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        return dict(explanation.as_list())

    def batch_explain(
        self,
        X: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> List[Dict[str, float]]:
        """Explain multiple instances.

        Args:
            X: Array of shape ``(n_samples, n_features)``.
            predict_fn: Model prediction function.
            num_features: Features per explanation.
            num_samples: LIME perturbation samples.

        Returns:
            List of explanation dicts, one per sample.
        """
        return [
            self.explain_instance(row, predict_fn, num_features, num_samples)
            for row in X
        ]


class ModelExplainabilityReport:
    """Convenience class that combines SHAP and LIME explanations."""

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        task: str = "regression",
    ) -> None:
        """Initialise the combined report.

        Args:
            model: Fitted model.
            X_train: Training data (background for SHAP/LIME).
            feature_names: Feature names.
            task: ``"regression"`` or ``"classification"``.
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task = task
        self._shap = SHAPExplainer(model, background_data=X_train)
        self._lime = LIMEExplainer(X_train, feature_names=feature_names, mode=task)

    def global_shap_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Return global SHAP importance for *X*.

        Args:
            X: Evaluation data.
        """
        return self._shap.global_importance(X, self.feature_names)

    def local_lime_explanation(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> Dict[str, float]:
        """Return a local LIME explanation for *instance*.

        Args:
            instance: Single sample (1-D).
            num_features: Number of top features to return.
        """
        predict_fn = (
            self.model.predict_proba
            if self.task == "classification" and hasattr(self.model, "predict_proba")
            else self.model.predict
        )
        return self._lime.explain_instance(instance, predict_fn, num_features=num_features)
