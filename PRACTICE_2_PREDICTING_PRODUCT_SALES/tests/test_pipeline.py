import numpy as np
import pandas as pd

from scripts.pipeline import build_features, chronological_split, evaluate


def test_build_features_creates_calendar_features():
    sample = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-02"]),
        "sales": [10.0, 20.0],
        "store_nbr": [1, 1],
        "family": ["GROCERY I", "GROCERY I"],
        "onpromotion": [0, 2],
        "city": ["Quito", "Quito"],
        "state": ["Pichincha", "Pichincha"],
        "type": ["D", "D"],
        "cluster": [13, 13],
        "dcoilwtico": [50.0, np.nan],
        "is_holiday": [0, 0]
    })
    X, y = build_features(sample)
    assert "day_of_week" in X.columns
    assert len(y) == 2


def test_chronological_split_keeps_order():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    X_train, X_valid, y_train, y_valid = chronological_split(X, y, 0.2)
    assert X_train.x.max() < X_valid.x.min()
    assert y_train.max() < y_valid.min()


def test_chronological_split_never_splits_same_date():
    dates = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02",
        "2024-01-02", "2024-01-03", "2024-01-03",
        "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"
    ]))
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    X_train, X_valid, _, _ = chronological_split(X, y, 0.2, dates=dates)
    assert len(X_train) + len(X_valid) == len(X)
    assert dates.iloc[X_train.index].max() < dates.iloc[X_valid.index].min()


def test_evaluate_returns_three_metrics():
    result = evaluate([1, 2, 3], [1, 3, 2])
    assert set(result) == {"MAE", "MSE", "RMSE"}
    assert result["MAE"] > 0
