"""
tasks/machine_learning/feature_engineering.py
-----------------------------------------------
Generic feature engineering helpers for tabular competition tasks.

These are building-block transforms applied BEFORE handing data to TabularPipeline.
They are intentionally simple and fast — competition time is expensive.

Usage:
    from tasks.machine_learning.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    X_train_clean = fe.fit_transform(X_train)
    X_test_clean  = fe.transform(X_test)
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateful feature engineering transformer compatible with sklearn pipelines.

    Steps applied (in order):
        1. Drop near-zero-variance columns
        2. Impute missing values (median for numeric, mode for categorical)
        3. Encode categorical columns (label encoding)
        4. Optionally create interaction features (top-N numeric pairs)
        5. Clip outliers (IQR method — optional)

    All state (medians, modes, encoders, column lists) is fit on train,
    applied identically on test/inference to prevent data leakage.
    """

    def __init__(
        self,
        max_cardinality: int = 50,
        create_interactions: bool = False,
        n_interaction_pairs: int = 10,
        clip_outliers: bool = False,
        variance_threshold: float = 0.0,
    ):
        """
        Args:
            max_cardinality:      Drop categorical columns with more unique values than this
            create_interactions:  Multiply top-N numeric feature pairs
            n_interaction_pairs:  How many interaction pairs to create
            clip_outliers:        Clip values beyond 1.5*IQR (conservative)
            variance_threshold:   Drop numeric columns with variance <= this
        """
        self.max_cardinality = max_cardinality
        self.create_interactions = create_interactions
        self.n_interaction_pairs = n_interaction_pairs
        self.clip_outliers = clip_outliers
        self.variance_threshold = variance_threshold

        # Learned state
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._drop_cols: list[str] = []
        self._medians: dict = {}
        self._modes: dict = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._iqr_bounds: dict[str, tuple] = {}
        self._interaction_pairs: list[tuple[str, str]] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        X = X.copy()

        # 1. Identify column types
        self._numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

        # 2. Drop near-zero variance numeric columns
        self._drop_cols = [
            c for c in self._numeric_cols
            if X[c].var() <= self.variance_threshold
        ]
        self._numeric_cols = [c for c in self._numeric_cols if c not in self._drop_cols]

        # 3. Drop high-cardinality categoricals (they'll hurt more than help)
        high_card = [c for c in cat_cols if X[c].nunique() > self.max_cardinality]
        self._drop_cols += high_card
        self._categorical_cols = [c for c in cat_cols if c not in high_card]

        # 4. Learn imputation values
        for c in self._numeric_cols:
            self._medians[c] = X[c].median()
        for c in self._categorical_cols:
            self._modes[c] = X[c].mode().iloc[0] if not X[c].mode().empty else "MISSING"

        # 5. Learn label encoders
        for c in self._categorical_cols:
            le = LabelEncoder()
            le.fit(X[c].fillna(self._modes[c]).astype(str))
            self._label_encoders[c] = le

        # 6. Learn IQR bounds
        if self.clip_outliers:
            for c in self._numeric_cols:
                q1, q3 = X[c].quantile(0.25), X[c].quantile(0.75)
                iqr = q3 - q1
                self._iqr_bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # 7. Select interaction pairs (most correlated pairs with target, if y provided)
        if self.create_interactions and len(self._numeric_cols) >= 2:
            if y is not None:
                y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
                corrs = X[self._numeric_cols].corrwith(y_series).abs()
                top = corrs.nlargest(min(len(self._numeric_cols), 6)).index.tolist()
            else:
                top = self._numeric_cols[:6]
            pairs = [(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]
            self._interaction_pairs = pairs[:self.n_interaction_pairs]

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = X.copy()

        # Drop unwanted columns (ignore missing in test set)
        existing_drops = [c for c in self._drop_cols if c in X.columns]
        X = X.drop(columns=existing_drops, errors="ignore")

        # Impute numeric
        for c in self._numeric_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self._medians.get(c, 0))

        # Clip outliers
        if self.clip_outliers:
            for c, (lo, hi) in self._iqr_bounds.items():
                if c in X.columns:
                    X[c] = X[c].clip(lower=lo, upper=hi)

        # Impute + encode categoricals
        for c in self._categorical_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self._modes.get(c, "MISSING")).astype(str)
                le = self._label_encoders[c]
                # Handle unseen labels by mapping them to the most frequent class
                known = set(le.classes_)
                X[c] = X[c].apply(lambda v: v if v in known else le.classes_[0])
                X[c] = le.transform(X[c])

        # Interaction features
        for (c1, c2) in self._interaction_pairs:
            if c1 in X.columns and c2 in X.columns:
                X[f"{c1}_x_{c2}"] = X[c1] * X[c2]

        return X

    def get_feature_names_out(self) -> list[str]:
        base = self._numeric_cols + self._categorical_cols
        interaction_names = [f"{c1}_x_{c2}" for c1, c2 in self._interaction_pairs]
        return base + interaction_names


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    X = data.data
    fe = FeatureEngineer(create_interactions=True, clip_outliers=True)
    X_out = fe.fit_transform(X)
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {X_out.shape}")
    print(f"Features: {fe.get_feature_names_out()[:5]} ...")
