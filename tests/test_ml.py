"""
tests/test_ml.py
-----------------
Unit tests for the machine_learning task module.

Run:
    pytest tests/test_ml.py -v
"""

import numpy as np
import pandas as pd
import pytest

from tasks.machine_learning.baseline import TabularPipeline
from tasks.machine_learning.feature_engineering import FeatureEngineer


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def classification_data():
    """Small synthetic classification dataset."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "f0": rng.normal(0, 1, 100),
        "f1": rng.normal(1, 2, 100),
        "f2": rng.uniform(-1, 1, 100),
    })
    y = pd.Series((X["f0"] + X["f1"] > 1).astype(int))
    return X, y


@pytest.fixture
def regression_data():
    """Small synthetic regression dataset."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = pd.Series(X["a"] * 2.0 + X["b"] * 0.5 + rng.normal(scale=0.1, size=100))
    return X, y


@pytest.fixture
def dirty_data():
    """Dataset with missing values and a categorical column."""
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "num_a": rng.normal(size=n),
        "num_b": rng.uniform(0, 1, size=n),
        "cat_x": rng.choice(["red", "blue", "green"], size=n),
    })
    # Inject some NaN
    df.loc[rng.choice(n, 10, replace=False), "num_a"] = np.nan
    df.loc[rng.choice(n, 5,  replace=False), "cat_x"] = None
    return df


# ===========================================================================
# TabularPipeline — Classification
# ===========================================================================

def test_classification_fit_predict(classification_data):
    X, y = classification_data
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X, y, cv_folds=3, verbose=False)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_classification_predict_proba(classification_data):
    X, y = classification_data
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X, y, cv_folds=3, verbose=False)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classification_evaluate(classification_data):
    X, y = classification_data
    split = 80
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X.iloc[:split], y.iloc[:split], cv_folds=3, verbose=False)
    metrics = pipeline.evaluate(X.iloc[split:], y.iloc[split:])
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_classification_with_string_labels():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    y = pd.Series(["cat", "dog"] * 30)
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X, y, cv_folds=2, verbose=False)
    preds = pipeline.predict(X)
    assert set(preds).issubset({"cat", "dog"})


def test_predict_before_fit_raises():
    pipeline = TabularPipeline(task="classification")
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.predict(pd.DataFrame({"x": [1.0]}))


# ===========================================================================
# TabularPipeline — Regression
# ===========================================================================

def test_regression_fit_predict(regression_data):
    X, y = regression_data
    pipeline = TabularPipeline(task="regression")
    pipeline.fit(X, y, cv_folds=3, verbose=False)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert preds.dtype in [np.float32, np.float64]


def test_regression_evaluate(regression_data):
    X, y = regression_data
    pipeline = TabularPipeline(task="regression")
    pipeline.fit(X, y, cv_folds=3, verbose=False)
    metrics = pipeline.evaluate(X, y)
    assert "mae" in metrics and "rmse" in metrics
    assert metrics["mae"] >= 0


def test_regression_no_proba(regression_data):
    X, y = regression_data
    pipeline = TabularPipeline(task="regression")
    pipeline.fit(X, y, cv_folds=2, verbose=False)
    with pytest.raises(ValueError):
        pipeline.predict_proba(X)


# ===========================================================================
# TabularPipeline — Persistence
# ===========================================================================

def test_save_load_roundtrip(classification_data, tmp_path):
    X, y = classification_data
    pipeline = TabularPipeline(task="classification")
    pipeline.fit(X, y, cv_folds=2, verbose=False)
    preds_before = pipeline.predict(X)

    save_path = str(tmp_path / "model.pkl")
    pipeline.save(save_path)

    loaded = TabularPipeline.load(save_path)
    preds_after = loaded.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)


# ===========================================================================
# FeatureEngineer
# ===========================================================================

def test_feature_engineer_basic(dirty_data):
    fe = FeatureEngineer()
    result = fe.fit_transform(dirty_data)
    assert result.isnull().sum().sum() == 0, "No NaN values should remain after transform"
    assert result.shape[0] == dirty_data.shape[0]


def test_feature_engineer_no_data_leakage(dirty_data):
    """Encoders fitted on train must be applied on test without refitting."""
    fe = FeatureEngineer()
    fe.fit(dirty_data)
    result = fe.transform(dirty_data.head(20))
    assert result.shape[0] == 20


def test_feature_engineer_interaction_features(classification_data):
    X, y = classification_data
    fe = FeatureEngineer(create_interactions=True, n_interaction_pairs=3)
    result = fe.fit_transform(X, y=y)
    # Should have more columns than the original
    assert result.shape[1] > X.shape[1]


def test_feature_engineer_unseen_categories():
    """Unseen categories in test set must not cause errors."""
    train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    test  = pd.DataFrame({"cat": ["a", "UNSEEN", "b"]})
    fe = FeatureEngineer()
    fe.fit(train)
    result = fe.transform(test)
    assert result.shape[0] == 3
    assert result.isnull().sum().sum() == 0
