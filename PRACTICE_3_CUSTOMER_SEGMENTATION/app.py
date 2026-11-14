"""Flask demo for Practice 3: customer clustering and image segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
IMAGE_OUTPUTS = OUTPUTS / "segmented_images"
ALLOWED_K = {3, 5, 8}

app = Flask(__name__)


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def dashboard_data():
    summary_data = read_json(OUTPUTS / "metrics" / "run_summary.json")
    summary = summary_data[0] if isinstance(summary_data, list) and summary_data else {}
    profiles_path = OUTPUTS / "metrics" / "cluster_profiles.csv"
    comparison_path = OUTPUTS / "metrics" / "clustering_comparison.csv"
    profiles = pd.read_csv(profiles_path).to_dict(orient="records") if profiles_path.exists() else []
    comparison = pd.read_csv(comparison_path).fillna("").to_dict(orient="records") if comparison_path.exists() else []
    return summary, profiles, comparison


@app.get("/")
def index():
    try:
        summary, profiles, comparison = dashboard_data()
        selected_k = request.args.get("k", default=5, type=int)
        if selected_k not in ALLOWED_K:
            selected_k = 5
        image_exists = (IMAGE_OUTPUTS / f"comparison_k{selected_k}.png").exists()
        missing = []
        if not summary:
            missing.append("outputs/metrics/run_summary.json")
        if not profiles:
            missing.append("outputs/metrics/cluster_profiles.csv")
        if not image_exists:
            missing.append(f"outputs/segmented_images/comparison_k{selected_k}.png")
        return render_template(
            "index.html",
            summary=summary,
            profiles=profiles,
            comparison=comparison,
            selected_k=selected_k,
            image_exists=image_exists,
            missing=missing,
        )
    except Exception as exc:
        app.logger.exception("Không thể tải dữ liệu dashboard")
        return render_template(
            "index.html", summary={}, profiles=[], comparison=[], selected_k=5,
            image_exists=False, missing=[str(exc)]
        ), 500


@app.get("/api/dashboard")
def api_dashboard():
    summary, profiles, comparison = dashboard_data()
    return jsonify({"summary": summary, "profiles": profiles, "comparison": comparison})


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUTS, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
