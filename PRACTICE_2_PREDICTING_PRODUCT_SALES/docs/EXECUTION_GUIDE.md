# Execution guide

Run commands from `PRACTICE_2_PREDICTING_PRODUCT_SALES/`.

1. Install: `pip install -r requirements.txt`
2. Place Kaggle CSV files in `data/raw/`.
3. EDA: `python -m scripts.run_eda`
4. Model pipeline: `python -m scripts.run_pipeline`
5. Tests: `python -m pytest -q`

EDA outputs are written to `outputs/eda/`; model metrics and predictions are written to `outputs/metrics/` and `outputs/predictions/`.

The data files are excluded from Git to avoid committing large competition datasets. Keep original raw files unchanged.
