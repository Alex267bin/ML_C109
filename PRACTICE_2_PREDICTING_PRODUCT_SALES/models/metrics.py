"""Các chỉ số đánh giá hồi quy tự cài đặt bằng NumPy."""
from __future__ import annotations

from typing import Any
import numpy as np

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


