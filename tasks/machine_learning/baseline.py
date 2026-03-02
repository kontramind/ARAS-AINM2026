"""
tasks/machine_learning/baseline.py
-----------------------------------
A competition-ready tabular ML pipeline.

Design goals:
  - Start with a baseline in < 5 minutes from new data
  - Run a multi-model race and pick the best automatically
  - Expose a clean fit / predict / evaluate interface
  - Serialize / load artifacts with a single call (fast swap at competition time)

Usage:
    from tasks.machine_learning import TabularPipeline

    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = pipeline.evaluate(X_val, y_val)
    pipeline.save("models/tabular_v1.pkl")

    # Later, on the API server:
    pipeline = TabularPipeline.load("models/tabular_v1.pkl")
    preds = pipeline.predict(X_new)
"""

import os
import warnings
import time
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Optional heavy boosting libs — gracefully degrade if not installed
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


class TabularPipeline:
    """
    End-to-end tabular ML pipeline with automatic model selection.

    Supports:
        - Binary / multi-class classification
        - Regression

    Model roster (all trained + evaluated on cross-validation):
        - RandomForest (always available)
        - GradientBoosting (always available)
        - XGBoost (if installed)
        - LightGBM (if installed)
        - Logistic Regression / Ridge (fast baseline)

    The winner is stored as `self.best_model`.
    """

    def __init__(self, task: Literal["classification", "regression"] = "classification"):
        self.task = task
        self.best_model: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._feature_names: list[str] = []
        self._cv_results: dict = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray,
            cv_folds: int = 5, verbose: bool = True) -> "TabularPipeline":
        """
        Train all candidate models, compare with CV, keep the best.

        Args:
            X:        Feature matrix (DataFrame or ndarray)
            y:        Target vector
            cv_folds: Number of cross-validation folds
            verbose:  Print race results

        Returns:
            self (chainable)
        """
        X = self._to_frame(X)
        self._feature_names = list(X.columns)

        # Encode string labels for classification
        if self.task == "classification":
            if y.dtype == object or isinstance(y.iloc[0] if hasattr(y, "iloc") else y[0], str):
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)

        candidates = self._build_candidates()

        if verbose:
            print(f"\n🏁 Starting model race — task={self.task}, cv_folds={cv_folds}")
            print(f"   Candidates: {list(candidates.keys())}\n")

        scoring = "f1_weighted" if self.task == "classification" else "neg_mean_absolute_error"
        best_score = -np.inf
        best_name = None

        for name, model in candidates.items():
            t0 = time.time()
            pipeline = self._wrap_in_pipeline(model)
            scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            elapsed = time.time() - t0
            mean_score = scores.mean()
            self._cv_results[name] = {"mean": mean_score, "std": scores.std(), "time_s": elapsed}

            if verbose:
                print(f"   {name:<20} score={mean_score:.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

            if mean_score > best_score:
                best_score = mean_score
                best_name = name

        # Refit the winner on the entire training set
        self.best_model = self._wrap_in_pipeline(candidates[best_name])
        self.best_model.fit(X, y)

        if verbose:
            print(f"\n🏆 Winner: {best_name} (score={best_score:.4f})")

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Run inference. Returns decoded labels for classification."""
        self._assert_fitted()
        X = self._to_frame(X)
        preds = self.best_model.predict(X)
        if self.task == "classification" and self.label_encoder is not None:
            preds = self.label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities (classification only)."""
        self._assert_fitted()
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        X = self._to_frame(X)
        return self.best_model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame | np.ndarray,
                 y: pd.Series | np.ndarray) -> dict:
        """Compute and print evaluation metrics on a held-out set."""
        self._assert_fitted()
        preds = self.predict(X)

        if self.task == "classification":
            # Decode true labels if needed
            if self.label_encoder is not None and (
                y.dtype == object or isinstance(y.iloc[0] if hasattr(y, "iloc") else y[0], str)
            ):
                y_enc = self.label_encoder.transform(y)
            else:
                y_enc = y
            preds_enc = (self.label_encoder.transform(preds)
                         if self.label_encoder is not None else preds)

            acc = accuracy_score(y_enc, preds_enc)
            f1 = f1_score(y_enc, preds_enc, average="weighted")
            metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}

            try:
                proba = self.predict_proba(X)
                n_classes = proba.shape[1]
                roc = roc_auc_score(y_enc, proba if n_classes > 2 else proba[:, 1],
                                    multi_class="ovr" if n_classes > 2 else "raise")
                metrics["roc_auc"] = round(roc, 4)
            except Exception:
                pass

        else:
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4)}

        print("📊 Evaluation:", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the entire pipeline to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TabularPipeline":
        """Load a previously saved pipeline."""
        pipeline = joblib.load(path)
        print(f"📂 Pipeline loaded ← {path}")
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(self) -> dict:
        candidates = {}
        if self.task == "classification":
            candidates["RandomForest"] = RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
            candidates["GradientBoosting"] = GradientBoostingClassifier(
                n_estimators=200, random_state=RANDOM_SEED)
            candidates["LogisticRegression"] = LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED)
            if _HAS_XGB:
                candidates["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=200, random_state=RANDOM_SEED,
                    eval_metric="logloss", verbosity=0)
            if _HAS_LGB:
                candidates["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=200, random_state=RANDOM_SEED, verbose=-1)
        else:
            candidates["RandomForest"] = RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
            candidates["GradientBoosting"] = GradientBoostingRegressor(
                n_estimators=200, random_state=RANDOM_SEED)
            candidates["Ridge"] = Ridge()
            if _HAS_XGB:
                candidates["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=200, random_state=RANDOM_SEED, verbosity=0)
            if _HAS_LGB:
                candidates["LightGBM"] = lgb.LGBMRegressor(
                    n_estimators=200, random_state=RANDOM_SEED, verbose=-1)
        return candidates

    def _wrap_in_pipeline(self, model) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    @staticmethod
    def _to_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return X.reset_index(drop=True)

    def _assert_fitted(self):
        if self.best_model is None:
            raise RuntimeError("Pipeline is not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    split = int(0.8 * len(X))
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X.iloc[:split], y.iloc[:split])
    metrics = pipeline.evaluate(X.iloc[split:], y.iloc[split:])
    pipeline.save("models/smoke_test.pkl")
