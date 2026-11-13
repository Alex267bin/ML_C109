"""Quy trình học máy dự đoán doanh số sản phẩm hằng ngày của Favorita."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
    return frame.loc[usable, columns].copy(), target.loc[usable].astype("float32")


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = 0.2,
    dates: pd.Series | None = None,
):
    """Split on a date boundary so one calendar day cannot appear in both sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Tỷ lệ tập kiểm định phải nằm trong khoảng từ 0 đến 1.")

    if dates is None:
        # Compatibility fallback for small examples without a date column.
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


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])


def evaluate(y_true, predictions) -> dict:
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse))
    }


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    max_train_rows: int = 250_000,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = load_data(data_dir)
    LOGGER.info("Đã tải %s dòng dữ liệu và %s cột", *data.shape)
    X, y = build_features(data)
    X_train, X_valid, y_train, y_valid = chronological_split(
        X, y, validation_fraction=0.2, dates=data.loc[X.index, "date"]
    )

    # Keep the latest validation period intact. Sample only from the earlier
    # training period to bound memory and training time in Codespaces.
    if max_train_rows > 0 and len(X_train) > max_train_rows:
        sampled = X_train.sample(n=max_train_rows, random_state=42).sort_index()
        y_train = y_train.loc[sampled.index]
        X_train = sampled
        LOGGER.info("Số dòng dùng để huấn luyện: %s (được lấy mẫu từ giai đoạn huấn luyện theo thời gian)", len(X_train))
    LOGGER.info("Số dòng kiểm định: %s", len(X_valid))

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=30, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, max_depth=16, min_samples_leaf=10,
            max_features=0.8, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.08, max_depth=3,
            min_samples_leaf=30, subsample=0.8, random_state=42
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows, fitted = [], {}
    for name, estimator in candidates.items():
        LOGGER.info("Đang huấn luyện mô hình: %s", name)
        model = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("regressor", estimator)
        ])
        model.fit(X_train, y_train)
        predictions = np.maximum(0, model.predict(X_valid))
        rows.append({"Mô hình": name, **evaluate(y_valid, predictions)})
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("RMSE")
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, model_dir / "best_model.joblib")

    best_predictions = np.maximum(0, best_model.predict(X_valid))
    pd.DataFrame({
        "doanh_so_thuc_te": y_valid.to_numpy(),
        "doanh_so_du_doan": best_predictions
    }).to_csv(predictions_dir / "validation_predictions.csv", index=False)

    summary = {
        "mo_hinh_tot_nhat": best_name,
        "tieu_chi_lua_chon": "RMSE",
        "so_dong_huan_luyen": int(len(y_train)),
        "so_dong_kiem_dinh": int(len(y_valid)),
        "ngay_bat_dau_kiem_dinh": str(data.loc[X_valid.index, "date"].min().date()),
        "ngay_ket_thuc_kiem_dinh": str(data.loc[X_valid.index, "date"].max().date()),
        "cac_chi_so": evaluate(y_valid, best_predictions),
    }
    (metrics_dir / "best_model.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Mô hình tốt nhất: %s", best_name)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá các mô hình dự đoán doanh số sản phẩm.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/saved_models"))
    parser.add_argument(
        "--max-train-rows", type=int, default=250_000,
        help="Số dòng tối đa được lấy mẫu từ giai đoạn huấn luyện theo thời gian; nhập 0 để dùng toàn bộ dữ liệu."
    )
    args = parser.parse_args()
    result = run_pipeline(args.data_dir, args.output_dir, args.model_dir, args.max_train_rows)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
