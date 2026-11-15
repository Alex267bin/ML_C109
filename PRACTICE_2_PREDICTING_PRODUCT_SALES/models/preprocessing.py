"""Tiền xử lý dữ liệu bằng NumPy và Pandas, không dùng scikit-learn."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

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


