"""Ensemble methods combining XGBoost, CatBoost, and LightGBM.

All heavy ML-library imports are deferred inside functions so the module
loads successfully even when those packages are not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GradientBoostingEnsemble:
    """Ensemble of gradient-boosted tree models for trading signal generation.

    Uses XGBoost, LightGBM, and CatBoost as base learners and combines
    their predictions via a weighted average or a meta-learner.
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_catboost: bool = True,
        ensemble_method: str = "weighted_average",
        xgb_params: Optional[Dict[str, Any]] = None,
        lgbm_params: Optional[Dict[str, Any]] = None,
        cat_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the ensemble configuration.

        Args:
            use_xgboost: Include an XGBoost model in the ensemble.
            use_lightgbm: Include a LightGBM model in the ensemble.
            use_catboost: Include a CatBoost model in the ensemble.
            ensemble_method: ``"weighted_average"`` or ``"stacking"``.
            xgb_params: Override parameters for XGBoost.
            lgbm_params: Override parameters for LightGBM.
            cat_params: Override parameters for CatBoost.
        """
        self.use_xgboost = use_xgboost
        self.use_lightgbm = use_lightgbm
        self.use_catboost = use_catboost
        self.ensemble_method = ensemble_method
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.cat_params = cat_params or {}
        self._models: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._meta_model: Any = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_xgb_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            **self.xgb_params,
        }

    def _default_lgbm_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
            **self.lgbm_params,
        }

    def _default_cat_params(self) -> Dict[str, Any]:
        return {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "random_seed": 42,
            "verbose": 0,
            **self.cat_params,
        }

    def _build_models(self, task: str = "regression") -> None:
        """Instantiate the enabled base learners.

        Args:
            task: ``"regression"`` or ``"classification"``.
        """
        if self.use_xgboost:
            try:
                import xgboost as xgb  # noqa: PLC0415

                cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
                self._models["xgboost"] = cls(**self._default_xgb_params())
                logger.info("XGBoost model initialised.")
            except ImportError:
                logger.warning("xgboost not installed; skipping.")

        if self.use_lightgbm:
            try:
                import lightgbm as lgb  # noqa: PLC0415

                cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
                self._models["lightgbm"] = cls(**self._default_lgbm_params())
                logger.info("LightGBM model initialised.")
            except ImportError:
                logger.warning("lightgbm not installed; skipping.")

        if self.use_catboost:
            try:
                from catboost import CatBoostRegressor, CatBoostClassifier  # noqa: PLC0415

                cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
                self._models["catboost"] = cls(**self._default_cat_params())
                logger.info("CatBoost model initialised.")
            except ImportError:
                logger.warning("catboost not installed; skipping.")

        if not self._models:
            raise RuntimeError("No gradient-boosting libraries are available.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "GradientBoostingEnsemble":
        """Fit all base learners.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
            task: ``"regression"`` or ``"classification"``.
            eval_set: Optional ``(X_val, y_val)`` for early stopping.

        Returns:
            ``self`` for method chaining.
        """
        self._build_models(task)
        for name, model in self._models.items():
            logger.info("Fitting %s …", name)
            if eval_set and name == "xgboost":
                model.fit(X, y, eval_set=[eval_set], verbose=False)
            elif eval_set and name == "lightgbm":
                model.fit(X, y, eval_set=[eval_set])
            else:
                model.fit(X, y)
        # Equal weights by default
        n = len(self._models)
        self._weights = {name: 1.0 / n for name in self._models}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Produce ensemble predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Prediction array of shape ``(n_samples,)``.
        """
        if not self._models:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        preds = np.array([model.predict(X) for model in self._models.values()])
        weights = np.array([self._weights[n] for n in self._models])
        return np.average(preds, axis=0, weights=weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Produce class probability estimates (classification only).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        if not self._models:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        proba_list = [
            model.predict_proba(X)
            for name, model in self._models.items()
            if hasattr(model, "predict_proba")
        ]
        if not proba_list:
            raise RuntimeError("No classifiers available for predict_proba.")
        return np.mean(proba_list, axis=0)

    def feature_importance(self) -> Dict[str, np.ndarray]:
        """Return per-model feature importances.

        Returns:
            Dictionary mapping model name to importance array.
        """
        importances: Dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
        return importances

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Manually override ensemble weights.

        Args:
            weights: Mapping from model name to non-negative weight.
        """
        total = sum(weights.values())
        self._weights = {k: v / total for k, v in weights.items()}


class StackingEnsemble(GradientBoostingEnsemble):
    """Extension of :class:`GradientBoostingEnsemble` using a meta-learner.

    Base-learner out-of-fold predictions are used to train a linear
    meta-learner (Ridge regression or Logistic regression).
    """

    def __init__(self, n_folds: int = 5, **kwargs: Any) -> None:
        """Initialise the stacking ensemble.

        Args:
            n_folds: Number of cross-validation folds for OOF generation.
            **kwargs: Forwarded to :class:`GradientBoostingEnsemble`.
        """
        super().__init__(ensemble_method="stacking", **kwargs)
        self.n_folds = n_folds

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "StackingEnsemble":
        """Fit base learners via OOF then train the meta-model.

        Args:
            X: Training features.
            y: Targets.
            task: ``"regression"`` or ``"classification"``.
            eval_set: Unused; kept for API compatibility.

        Returns:
            ``self``.
        """
        try:
            from sklearn.model_selection import KFold  # noqa: PLC0415
            from sklearn.linear_model import Ridge, LogisticRegression  # noqa: PLC0415
        except ImportError as exc:
            logger.error("scikit-learn required for stacking: %s", exc)
            raise

        self._build_models(task)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        oof = np.zeros((len(X), len(self._models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            for col, (name, model) in enumerate(self._models.items()):
                model.fit(X_tr, y_tr)
                oof[val_idx, col] = model.predict(X_val)
            logger.info("Stacking fold %d/%d done.", fold_idx + 1, self.n_folds)

        # Refit base models on full data
        for model in self._models.values():
            model.fit(X, y)

        # Train meta-model
        if task == "regression":
            self._meta_model = Ridge()
        else:
            self._meta_model = LogisticRegression(max_iter=1000)
        self._meta_model.fit(oof, y)
        logger.info("Meta-learner trained.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Produce stacked predictions.

        Args:
            X: Feature matrix.

        Returns:
            Prediction array.
        """
        if self._meta_model is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        base_preds = np.column_stack([model.predict(X) for model in self._models.values()])
        return self._meta_model.predict(base_preds)
