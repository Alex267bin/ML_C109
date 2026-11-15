"""Gradient Boosting Regressor học tuần tự trên residual."""
from __future__ import annotations

import numpy as np
from .decision_tree import DecisionTreeRegressorCustom

class GradientBoostingRegressorCustom:
    """Gradient Boosting cho MSE: mỗi cây học residual của mô hình hiện tại."""

    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.05,
                 max_depth: int = 2, min_samples_leaf: int = 20,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        self.initial_prediction = float(np.mean(y))
        current = np.full(len(y), self.initial_prediction, dtype=float)
        self.trees = []
        for index in range(self.n_estimators):
            residual = y - current
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + index,
            ).fit(X, residual)
            current += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = np.full(len(X), self.initial_prediction, dtype=float)
        for tree in self.trees:
            result += self.learning_rate * tree.predict(X)
        return result
