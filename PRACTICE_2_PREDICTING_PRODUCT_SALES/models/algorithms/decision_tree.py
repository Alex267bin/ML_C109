"""Decision Tree Regressor tự cài đặt theo tiêu chí CART."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class _TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: Any = None
    right: Any = None


class DecisionTreeRegressorCustom:
    """Cây hồi quy CART; chọn ngưỡng theo mức giảm tổng bình phương sai số."""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20,
                 max_features: int | None = None, max_thresholds: int = 16,
                 random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if self.X.ndim != 2 or len(self.X) != len(self.y) or not len(self.y):
            raise ValueError("Dữ liệu huấn luyện không hợp lệ.")
        self.rng = np.random.default_rng(self.random_state)
        self.root = self._build(np.arange(len(self.y)), 0)
        del self.X, self.y, self.rng
        return self

    def _build(self, indices: np.ndarray, depth: int):
        target = self.y[indices]
        node = _TreeNode(float(np.mean(target)))
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        feature_count = self.X.shape[1]
        if feature_count == 0:
            return node
        count = self.max_features or feature_count
        features = self.rng.choice(feature_count, size=min(count, feature_count), replace=False)
        best_loss = float(np.sum((target - node.value) ** 2))
        best = None
        for feature in features:
            values = self.X[indices, feature]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            if len(unique) > self.max_thresholds + 1:
                quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, quantiles))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2
            for threshold in thresholds:
                mask = values <= threshold
                left_count = int(mask.sum())
                right_count = len(indices) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_y, right_y = target[mask], target[~mask]
                loss = float(np.sum((left_y - left_y.mean()) ** 2) +
                              np.sum((right_y - right_y.mean()) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best = (int(feature), float(threshold), indices[mask], indices[~mask])
        if best is None:
            return node
        node.feature, node.threshold = best[0], best[1]
        node.left = self._build(best[2], depth + 1)
        node.right = self._build(best[3], depth + 1)
        return node

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.root
        while node.feature >= 0:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in values], dtype=float)


