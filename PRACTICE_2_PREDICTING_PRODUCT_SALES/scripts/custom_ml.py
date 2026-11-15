"""Các thuật toán hồi quy tự cài đặt bằng NumPy; không phụ thuộc scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("y_true và y_pred phải cùng kích thước và không rỗng.")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


class NumpyPreprocessor:
    """Điền thiếu, chuẩn hóa cột số và One-Hot cột phân loại bằng NumPy."""

    def fit(self, frame: pd.DataFrame):
        self.columns = list(frame.columns)
        self.numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.categorical_columns = [c for c in self.columns if c not in self.numeric_columns]
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.modes: dict[str, str] = {}
        self.categories: dict[str, list[str]] = {}

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            self.medians[column] = median
            self.means[column] = mean
            self.scales[column] = scale if scale > 1e-12 else 1.0

        for column in self.categorical_columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            mode_values = values.mode()
            self.modes[column] = str(mode_values.iloc[0]) if not mode_values.empty else "__MISSING__"
            self.categories[column] = sorted(set(values.tolist()) | {"__UNKNOWN__"})
        self.feature_names = []
        self.feature_names.extend(self.numeric_columns)
        for column in self.categorical_columns:
            self.feature_names.extend(f"{column}={category}" for category in self.categories[column])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns"):
            raise RuntimeError("Cần gọi fit trước transform.")
        parts: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.medians[column])
            values = (values - self.means[column]) / self.scales[column]
            parts.append(values.reshape(-1, 1))
        for column in self.categorical_columns:
            values = frame[column].fillna(self.modes[column]).astype(str).to_numpy()
            categories = self.categories[column]
            known = set(categories)
            values = np.array([v if v in known else "__UNKNOWN__" for v in values], dtype=object)
            encoded = np.column_stack([(values == category).astype(float) for category in categories])
            parts.append(encoded)
        return np.hstack(parts).astype(np.float64) if parts else np.empty((len(frame), 0))

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class LinearRegressionGD:
    """Hồi quy tuyến tính huấn luyện bằng mini-batch Gradient Descent."""

    def __init__(self, learning_rate: float = 0.03, epochs: int = 40,
                 batch_size: int = 8192, l2: float = 1e-5, random_state: int = 42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or not len(y):
            raise ValueError("Dữ liệu huấn luyện không hợp lệ.")
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = float(np.mean(y))
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                indices = order[start:start + self.batch_size]
                xb, yb = X[indices], y[indices]
                error = xb @ self.weights + self.bias - yb
                self.weights -= self.learning_rate * (
                    (2.0 / len(indices)) * (xb.T @ error) + 2 * self.l2 * self.weights
                )
                self.bias -= self.learning_rate * 2.0 * float(np.mean(error))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.weights + self.bias


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
