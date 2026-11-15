"""Các chỉ số đánh giá hồi quy tự cài đặt bằng NumPy."""
from __future__ import annotations

from typing import Any
import numpy as np


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Tính MAE, MSE, RMSE và MAPE.

    MAPE bỏ qua các mẫu có giá trị thực tế bằng 0 vì phép chia cho 0
    không xác định. Nếu toàn bộ giá trị thực tế đều bằng 0, MAPE trả
    về NaN để thể hiện metric này không xác định trên tập dữ liệu đó.
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("y_true và y_pred phải cùng kích thước và không rỗng.")

    error = predicted - actual
    mse = float(np.mean(error ** 2))

    # MAPE chỉ tính trên các mẫu có giá trị thực tế khác 0.
    nonzero_mask = actual != 0
    if np.any(nonzero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (actual[nonzero_mask] - predicted[nonzero_mask])
                    / actual[nonzero_mask]
                )
            ) * 100
        )
    else:
        mape = float("nan")

    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": mape,
    }
