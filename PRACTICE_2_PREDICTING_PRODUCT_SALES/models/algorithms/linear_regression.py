"""Linear Regression huấn luyện bằng mini-batch Gradient Descent."""
from __future__ import annotations

import numpy as np
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
        self.target_mean = float(np.mean(y))
        self.target_scale = float(np.std(y))
        if self.target_scale <= 1e-12:
            self.target_scale = 1.0
        y_scaled = (y - self.target_mean) / self.target_scale
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = float(np.mean(y_scaled))
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                indices = order[start:start + self.batch_size]
                xb, yb = X[indices], y_scaled[indices]
                error = xb @ self.weights + self.bias - yb
                self.weights -= self.learning_rate * (
                    (2.0 / len(indices)) * (xb.T @ error) + 2 * self.l2 * self.weights
                )
                self.bias -= self.learning_rate * 2.0 * float(np.mean(error))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scaled = np.asarray(X, dtype=float) @ self.weights + self.bias
        return scaled * self.target_scale + self.target_mean


