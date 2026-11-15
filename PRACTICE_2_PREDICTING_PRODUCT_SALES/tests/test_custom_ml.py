import numpy as np
import pandas as pd

from models.preprocessing import NumpyPreprocessor
from models.metrics import regression_metrics
from models.algorithms import (
    NumpyPreprocessor,
    LinearRegressionGD,
    DecisionTreeRegressorCustom,
    RandomForestRegressorCustom,
    GradientBoostingRegressorCustom,
    regression_metrics,
)


def test_preprocessor_handles_missing_and_unknown_categories():
    train = pd.DataFrame({
        "number": [1.0, np.nan, 3.0],
        "category": ["A", "B", None],
    })
    processor = NumpyPreprocessor().fit(train)
    transformed = processor.transform(pd.DataFrame({
        "number": [2.0, np.nan],
        "category": ["C", "A"],
    }))
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


def test_linear_regression_learns_linear_relationship():
    x = ((np.arange(30, dtype=float) - 14.5) / 8.7).reshape(-1, 1)
    y = 4.0 * x[:, 0] + 7.0
    model = LinearRegressionGD(learning_rate=0.05, epochs=100, batch_size=10)
    model.fit(x, y)
    predictions = model.predict(x)
    assert np.mean(np.abs(predictions - y)) < 1.0


def test_decision_tree_learns_simple_step():
    x = np.arange(40, dtype=float).reshape(-1, 1)
    y = (x[:, 0] > 19).astype(float) * 10
    model = DecisionTreeRegressorCustom(max_depth=2, min_samples_leaf=2, max_thresholds=40)
    model.fit(x, y)
    assert np.mean(np.abs(model.predict(x) - y)) < 0.1


def test_random_forest_and_gradient_boosting_predict():
    x = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.sin(x[:, 0] / 5)
    forest = RandomForestRegressorCustom(
        n_estimators=3, max_depth=3, min_samples_leaf=2
    ).fit(x, y)
    boosting = GradientBoostingRegressorCustom(
        n_estimators=3, max_depth=2, min_samples_leaf=2
    ).fit(x, y)
    assert forest.predict(x[:5]).shape == (5,)
    assert boosting.predict(x[:5]).shape == (5,)


def test_metrics_are_computed_without_external_ml_library():
    result = regression_metrics([1, 2, 3], [1, 3, 2])
    assert result["MAE"] == 2 / 3
    assert result["MSE"] == 2 / 3
    assert np.isclose(result["RMSE"], np.sqrt(2 / 3))
