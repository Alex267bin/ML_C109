"""Exploratory data analysis utilities for the Favorita dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def inspect_dataset(data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find {train_path}. Add the Kaggle train.csv first.")

    df = pd.read_csv(train_path, parse_dates=["date"])
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "target_summary": df["sales"].describe().to_dict(),
        "families": int(df["family"].nunique()),
        "stores": int(df["store_nbr"].nunique()),
    }
    pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing_count": [int(v) for v in df.isna().sum()],
        "missing_percent": [float(v) for v in df.isna().mean() * 100],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).to_csv(output_dir / "data_quality.csv", index=False)

    monthly = df.assign(month=df["date"].dt.to_period("M").astype(str)).groupby("month")["sales"].sum()
    ax = monthly.plot(figsize=(12, 5), title="Total sales by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_sales.png", dpi=150)
    plt.close()

    family_sales = df.groupby("family")["sales"].sum().sort_values(ascending=False).head(15)
    ax = family_sales.sort_values().plot(kind="barh", figsize=(10, 7), title="Top 15 product families by total sales")
    ax.set_xlabel("Total sales")
    plt.tight_layout()
    plt.savefig(output_dir / "top_product_families.png", dpi=150)
    plt.close()

    pd.Series(report["target_summary"]).to_csv(output_dir / "sales_summary.csv", header=["value"])
    return report
