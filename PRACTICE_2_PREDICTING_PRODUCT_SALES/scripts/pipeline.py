"""End-to-end baseline pipeline for Favorita product sales forecasting."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

LOGGER = logging.getLogger("sales_pipeline")


def load_data(data_dir: Path) -> pd.DataFrame:
    required = ["train.csv", "stores.csv", "holidays_events.csv", "oil.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing)
            + ". Download the Kaggle Store Sales dataset and place files in data/raw/."
        )

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])

    data = train.merge(stores, on="store_nbr", how="left")
    data = data.merge(oil, on="date", how="left")
    holidays = holidays.copy()
    holidays["is_holiday"] = ~holidays["type"].isin(["Work Day"])
    holidays_daily = (holidays.groupby("date", as_index=False)["is_holiday"]
                      .max())
    data = data.merge(holidays_daily, on="date", how="left")
    data["is_holiday"] = data["is_holiday"].fillna(False).astype(int)
    return data


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = data.copy()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    if "onpromotion" not in frame:
        frame["onpromotion"] = 0
    if "dcoilwtico" not in frame:
        frame["dcoilwtico"] = np.nan

    target = pd.to_numeric(frame["sales"], errors="coerce")
    usable = target.notna() & (target >= 0)
    features = frame.loc[usable, [
        "store_nbr", "family", "onpromotion", "city", "state",
        "type", "cluster", "dcoilwtico", "is_holiday",
        "year", "month", "day", "day_of_week", "week_of_year", "is_weekend"
    ]].copy()
    return features, target.loc[usable].astype(float)


def chronological_split(X: pd.DataFrame, y: pd.Series, validation_fraction: float = 0.2):
    # Data is already ordered by date in the source train file.
    split_at = int(len(X) * (1 - validation_fraction))
    if split_at <= 0 or split_at >= len(X):
        raise ValueError("Not enough rows for a chronological train/validation split.")
    return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
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


def run_pipeline(data_dir: Path, output_dir: Path, model_dir: Path) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = load_data(data_dir)
    LOGGER.info("Loaded %s rows and %s columns", *data.shape)
    X, y = build_features(data)
    X_train, X_valid, y_train, y_valid = chronological_split(X, y)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=18, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=22, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        )
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows, fitted = [], {}
    for name, estimator in candidates.items():
        LOGGER.info("Training %s", name)
        model = Pipeline([("preprocessor", make_preprocessor(X_train)),
                          ("regressor", estimator)])
        model.fit(X_train, y_train)
        predictions = np.maximum(0, model.predict(X_valid))
        rows.append({"Model": name, **evaluate(y_valid, predictions)})
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("RMSE")
    comparison.to_csv(output_dir / "metrics" / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, model_dir / "best_model.joblib")

    best_predictions = np.maximum(0, best_model.predict(X_valid))
    pd.DataFrame({
        "actual_sales": y_valid.to_numpy(),
        "predicted_sales": best_predictions
    }).to_csv(output_dir / "predictions" / "validation_predictions.csv", index=False)

    summary = {
        "best_model": best_name,
        "selection_metric": "RMSE",
        "validation_rows": int(len(y_valid)),
        "metrics": evaluate(y_valid, best_predictions)
    }
    (output_dir / "metrics" / "best_model.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Best model: %s", best_name)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate product sales models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/saved_models"))
    args = parser.parse_args()
    print(run_pipeline(args.data_dir, args.output_dir, args.model_dir).to_string(index=False))


if __name__ == "__main__":
    main()
