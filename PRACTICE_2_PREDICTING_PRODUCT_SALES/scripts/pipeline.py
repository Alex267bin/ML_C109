"""Pipeline huấn luyện bốn thuật toán hồi quy tự cài đặt bằng NumPy."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.custom_ml import (
    NumpyPreprocessor,
    LinearRegressionGD,
    DecisionTreeRegressorCustom,
    RandomForestRegressorCustom,
    GradientBoostingRegressorCustom,
    regression_metrics,
)

LOGGER = logging.getLogger("sales_pipeline")


def load_data(data_dir: Path) -> pd.DataFrame:
    required = ["train.csv", "stores.csv", "holidays_events.csv", "oil.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu tệp dữ liệu: " + ", ".join(missing)
            + ". Hãy đặt các tệp dữ liệu Kaggle Store Sales vào thư mục data/raw/."
        )

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])

    data = train.merge(stores, on="store_nbr", how="left")
    data = data.merge(oil, on="date", how="left")
    holidays = holidays.copy()
    holidays["is_holiday"] = ~holidays["type"].isin(["Work Day"])
    holidays_daily = holidays.groupby("date", as_index=False)["is_holiday"].max()
    data = data.merge(holidays_daily, on="date", how="left")
    data["is_holiday"] = data["is_holiday"].fillna(False).astype("int8")
    return data.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = data.copy()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.day_of_week
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype("int16")
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype("int8")
    if "onpromotion" not in frame:
        frame["onpromotion"] = 0
    if "dcoilwtico" not in frame:
        frame["dcoilwtico"] = np.nan

    target = pd.to_numeric(frame["sales"], errors="coerce")
    usable = target.notna() & (target >= 0)
    columns = [
        "store_nbr", "family", "onpromotion", "city", "state", "type",
        "cluster", "dcoilwtico", "is_holiday", "year", "month", "day",
        "day_of_week", "week_of_year", "is_weekend"
    ]
    return frame.loc[usable, columns].copy(), target.loc[usable].astype("float64")


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = 0.2,
    dates: pd.Series | None = None,
):
    """Chia theo mốc ngày để một ngày lịch không xuất hiện ở cả hai tập."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Tỷ lệ tập kiểm định phải nằm trong khoảng từ 0 đến 1.")
    if dates is None:
        split_at = int(len(X) * (1 - validation_fraction))
        if split_at <= 0 or split_at >= len(X):
            raise ValueError("Không đủ dòng dữ liệu để chia theo thứ tự thời gian.")
        return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]

    date_values = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(date_values.dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Cần ít nhất hai ngày khác nhau để tạo tập kiểm định.")
    split_at = int(len(unique_dates) * (1 - validation_fraction))
    split_at = min(max(split_at, 1), len(unique_dates) - 1)
    validation_start = unique_dates[split_at]
    train_mask = date_values < validation_start
    valid_mask = ~train_mask
    return (
        X.loc[train_mask.to_numpy()], X.loc[valid_mask.to_numpy()],
        y.loc[train_mask.to_numpy()], y.loc[valid_mask.to_numpy()]
    )


def _sample_arrays(X: np.ndarray, y: np.ndarray, limit: int, seed: int = 42):
    if len(y) <= limit:
        return X, y
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(y), size=limit, replace=False))
    return X[indices], y[indices]


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    max_train_rows: int = 100_000,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = load_data(data_dir)
    LOGGER.info("Đã tải %s dòng dữ liệu và %s cột", *data.shape)
    X, y = build_features(data)
    X_train, X_valid, y_train, y_valid = chronological_split(
        X, y, validation_fraction=0.2, dates=data.loc[X.index, "date"]
    )

    if max_train_rows > 0 and len(X_train) > max_train_rows:
        X_train, y_train = _sample_arrays(
            X_train.to_numpy(dtype=object), y_train.to_numpy(), max_train_rows
        )
        # Khôi phục DataFrame để bộ tiền xử lý nhận biết cột phân loại.
        X_train = pd.DataFrame(X_train, columns=X.columns)
        LOGGER.info("Lấy mẫu %s dòng từ giai đoạn huấn luyện theo thời gian", len(X_train))

    preprocessor = NumpyPreprocessor().fit(X_train)
    train_matrix = preprocessor.transform(X_train)
    valid_matrix = preprocessor.transform(X_valid)
    train_target = y_train.to_numpy(dtype=float)
    valid_target = y_valid.to_numpy(dtype=float)
    LOGGER.info("Ma trận huấn luyện: %s; kiểm định: %s", train_matrix.shape, valid_matrix.shape)

    tree_X, tree_y = _sample_arrays(train_matrix, train_target, 50_000)
    forest_X, forest_y = _sample_arrays(train_matrix, train_target, 30_000)
    boost_X, boost_y = _sample_arrays(train_matrix, train_target, 30_000)

    candidates = [
        ("Linear Regression", LinearRegressionGD(
            learning_rate=0.03, epochs=40, batch_size=8192, random_state=42
        ), train_matrix, train_target),
        ("Decision Tree", DecisionTreeRegressorCustom(
            max_depth=12, min_samples_leaf=50, max_thresholds=12, random_state=42
        ), tree_X, tree_y),
        ("Random Forest", RandomForestRegressorCustom(
            n_estimators=12, max_depth=10, min_samples_leaf=50, random_state=42
        ), forest_X, forest_y),
        ("Gradient Boosting", GradientBoostingRegressorCustom(
            n_estimators=15, learning_rate=0.05, max_depth=2,
            min_samples_leaf=50, random_state=42
        ), boost_X, boost_y),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows, fitted = [], {}
    for name, estimator, fit_X, fit_y in candidates:
        LOGGER.info("Đang tự huấn luyện: %s", name)
        estimator.fit(fit_X, fit_y)
        predictions = np.maximum(0.0, estimator.predict(valid_matrix))
        rows.append({"Mô hình": name, **regression_metrics(valid_target, predictions)})
        fitted[name] = estimator

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["Mô hình"])
    best_model = fitted[best_name]
    with (model_dir / "best_model.pkl").open("wb") as file:
        pickle.dump({
            "preprocessor": preprocessor,
            "model": best_model,
            "model_name": best_name,
        }, file)

    best_predictions = np.maximum(0.0, best_model.predict(valid_matrix))
    pd.DataFrame({
        "doanh_so_thuc_te": valid_target,
        "doanh_so_du_doan": best_predictions
    }).to_csv(predictions_dir / "validation_predictions.csv", index=False)

    summary = {
        "mo_hinh_tot_nhat": best_name,
        "tieu_chi_lua_chon": "RMSE",
        "so_dong_huan_luyen": int(len(train_target)),
        "so_dong_kiem_dinh": int(len(valid_target)),
        "ngay_bat_dau_kiem_dinh": str(data.loc[X_valid.index, "date"].min().date()),
        "ngay_ket_thuc_kiem_dinh": str(data.loc[X_valid.index, "date"].max().date()),
        "cac_chi_so": regression_metrics(valid_target, best_predictions),
        "ghi_chu": "Các thuật toán và tiền xử lý được tự cài đặt bằng NumPy/Pandas.",
    }
    (metrics_dir / "best_model.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Mô hình tốt nhất: %s", best_name)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Tự huấn luyện và đánh giá bốn mô hình hồi quy.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/saved_models"))
    parser.add_argument(
        "--max-train-rows", type=int, default=100_000,
        help="Số dòng tối đa lấy mẫu từ giai đoạn huấn luyện; nhập 0 để dùng toàn bộ."
    )
    args = parser.parse_args()
    result = run_pipeline(args.data_dir, args.output_dir, args.model_dir, args.max_train_rows)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
