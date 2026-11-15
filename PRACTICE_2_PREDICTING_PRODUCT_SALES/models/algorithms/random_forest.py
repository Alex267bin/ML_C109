"""Random Forest Regressor dùng bootstrap và tập hợp cây quyết định."""
from __future__ import annotations

import numpy as np
from .decision_tree import DecisionTreeRegressorCustom

class RandomForestRegressorCustom:
    """Rừng cây CART dùng bootstrap và lấy trung bình dự đoán."""

    def __init__(self, n_estimators: int = 20, max_depth: int = 8,
                 min_samples_leaf: int = 20, max_features: int | None = None,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for index in range(self.n_estimators):
            sample = rng.integers(0, len(y), size=len(y))
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features or max(1, int(np.sqrt(X.shape[1]))),
                random_state=self.random_state + index,
            ).fit(X[sample], y[sample])
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


