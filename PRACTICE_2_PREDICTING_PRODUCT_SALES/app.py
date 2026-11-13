"""Local Flask demo for the trained Favorita product-sales model."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "saved_models" / "best_model.joblib"

app = Flask(__name__)


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy models/saved_models/best_model.joblib. "
            "Hãy chạy python -m scripts.run_pipeline để huấn luyện mô hình trước."
        )
    if not (DATA / "stores.csv").exists():
        raise FileNotFoundError("Thiếu data/raw/stores.csv.")
    model = joblib.load(MODEL_PATH)
    stores = pd.read_csv(DATA / "stores.csv")
    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    holidays["is_holiday"] = ~holidays["type"].isin(["Work Day"])
    holiday_days = set(
        holidays.groupby("date")["is_holiday"].max()
        .loc[lambda values: values].index.date
    )
    return model, stores, oil, holiday_days


def prediction_row(store_nbr: int, family: str, date_value: str, promotion: int):
    model, stores, oil, holiday_days = load_assets()
    store_matches = stores.loc[stores["store_nbr"] == store_nbr]
    if store_matches.empty:
        raise ValueError("Không tìm thấy cửa hàng đã chọn.")
    store = store_matches.iloc[0]
    date = pd.Timestamp(date_value)
    oil_before = oil.loc[oil["date"] <= date, "dcoilwtico"].dropna()
    oil_price = float(oil_before.iloc[-1]) if not oil_before.empty else float("nan")
    day_of_week = date.dayofweek
    row = pd.DataFrame([{
        "store_nbr": store_nbr,
        "family": family,
        "onpromotion": promotion,
        "city": store["city"],
        "state": store["state"],
        "type": store["type"],
        "cluster": store["cluster"],
        "dcoilwtico": oil_price,
        "is_holiday": int(date.date() in holiday_days),
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": day_of_week,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(day_of_week in (5, 6)),
    }])
    prediction = max(0.0, float(model.predict(row)[0]))
    return prediction


@app.get("/")
def index():
    try:
        _, stores, _, _ = load_assets()
        store_options = stores.sort_values("store_nbr").to_dict("records")
        families_path = DATA / "train.csv"
        families = sorted(pd.read_csv(families_path, usecols=["family"])["family"].dropna().unique())
        return render_template("index.html", stores=store_options, families=families)
    except Exception as exc:
        return render_template("index.html", stores=[], families=[], startup_error=str(exc))


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        store_nbr = int(payload["store_nbr"])
        family = str(payload["family"])
        date_value = str(payload["date"])
        promotion = int(payload.get("onpromotion", 0))
        if promotion < 0:
            raise ValueError("Số mặt hàng khuyến mãi không được âm.")
        prediction = prediction_row(store_nbr, family, date_value, promotion)
        return jsonify({"prediction": prediction, "unit": "đơn vị sản phẩm", "model": "Mô hình tốt nhất đã huấn luyện"})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Không thể tạo dự đoán. Kiểm tra dữ liệu và mô hình đã huấn luyện."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
