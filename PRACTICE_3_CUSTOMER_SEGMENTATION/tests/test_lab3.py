from pathlib import Path

import pandas as pd

from scripts.customer_clustering import FEATURES, load_customer_data
from scripts.image_segmentation import segment_image


def test_load_customer_data_accepts_expected_features(tmp_path: Path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({
        "Age": [20, 30, 40],
        "Annual Income (k$)": [20, 30, 40],
        "Spending Score (1-100)": [70, 50, 30],
    }).to_csv(path, index=False)
    df = load_customer_data(path)
    assert all(column in df.columns for column in FEATURES)
    assert len(df) == 3


def test_image_segmentation_creates_comparison(tmp_path: Path):
    output = segment_image(None, tmp_path, k=3, max_size=64)
    assert output.exists()
    assert output.stat().st_size > 0
    assert (tmp_path / "segmented_k3.png").exists()
