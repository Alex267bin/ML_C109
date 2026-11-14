"""Customer segmentation using K-Means, DBSCAN and Agglomerative Clustering."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_customer_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common column-name variants while preserving the source columns.
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    aliases = {
        "age": "Age",
        "annual income (k$)": "Annual Income (k$)",
        "annual income": "Annual Income (k$)",
        "spending score (1-100)": "Spending Score (1-100)",
        "spending score": "Spending Score (1-100)",
    }
    rename = {original: aliases[key] for key, original in normalized.items() if key in aliases}
    df = df.rename(columns=rename)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Required: {FEATURES}")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_clustering(data_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    df = load_customer_data(data_path)
    report = pd.DataFrame({
        "missing_values": df[FEATURES].isna().sum(),
        "outlier_count_IQR": [
            int(((df[c] < df[c].quantile(.25) - 1.5 * (df[c].quantile(.75)-df[c].quantile(.25))) |
                 (df[c] > df[c].quantile(.75) + 1.5 * (df[c].quantile(.75)-df[c].quantile(.25)))).sum())
            for c in FEATURES
        ],
    })
    report.to_csv(metrics_dir / "data_quality.csv")
    df = df.dropna(subset=FEATURES).copy()
    if len(df) < 3:
        raise ValueError("At least 3 complete customer records are required.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
    fig.tight_layout()
    fig.savefig(figures / "feature_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Gender" if "Gender" in df else None, ax=axes[0])
    sns.scatterplot(data=df, x=FEATURES[1], y=FEATURES[2], hue="Gender" if "Gender" in df else None, ax=axes[1])
    fig.tight_layout()
    fig.savefig(figures / "eda_scatterplots.png", dpi=160)
    plt.close(fig)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    scores = []
    max_k = min(10, len(df) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append({"algorithm": "KMeans", "parameters": f"k={k}", "n_clusters": k,
                       "silhouette": float(silhouette_score(X, labels)), "inertia": float(model.inertia_)})
    if not scores:
        raise ValueError("Not enough rows to evaluate K-Means.")

    # Experiment with density-based and hierarchical clustering too.
    for eps in (0.5, 0.8, 1.0, 1.2):
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        valid = labels != -1
        n_clusters = len(set(labels[valid]))
        if n_clusters >= 2 and valid.sum() > n_clusters:
            scores.append({"algorithm": "DBSCAN", "parameters": f"eps={eps}, min_samples=5",
                           "n_clusters": n_clusters,
                           "silhouette": float(silhouette_score(X[valid], labels[valid])), "inertia": np.nan})
    for k in (2, 3, 4, 5):
        if k < len(df):
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            scores.append({"algorithm": "Agglomerative", "parameters": f"k={k}", "n_clusters": k,
                           "silhouette": float(silhouette_score(X, labels)), "inertia": np.nan})

    score_df = pd.DataFrame(scores).sort_values("silhouette", ascending=False)
    score_df.to_csv(metrics_dir / "clustering_comparison.csv", index=False)
    kmeans_scores = score_df[score_df.algorithm == "KMeans"]
    best_row = kmeans_scores.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row.n_clusters)
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = final_model.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["PCA 1"], df["PCA 2"] = coords[:, 0], coords[:, 1]
    df.to_csv(output_dir / "customer_segments.csv", index=False)

    numeric_summary = df.groupby("Cluster")[FEATURES].mean().round(2)
    numeric_summary["Customer Count"] = df.groupby("Cluster").size()
    if "Gender" in df:
        gender = pd.crosstab(df["Cluster"], df["Gender"], normalize="index").round(3)
        gender.to_csv(metrics_dir / "gender_distribution_by_cluster.csv")
    numeric_summary.to_csv(metrics_dir / "cluster_profiles.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="PCA 1", y="PCA 2", hue="Cluster", palette="tab10", s=65, ax=ax)
    ax.set_title(f"Customer Segments (K-Means, k={best_k}) – PCA projection")
    fig.tight_layout()
    fig.savefig(figures / "customer_clusters_pca.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=kmeans_scores, x="n_clusters", y="silhouette", marker="o", ax=ax)
    ax.set_title("Silhouette score by K (K-Means)")
    fig.tight_layout()
    fig.savefig(figures / "silhouette_by_k.png", dpi=160)
    plt.close(fig)

    result = {"rows_used": len(df), "best_kmeans_k": best_k,
              "best_kmeans_silhouette": round(float(best_row.silhouette), 4),
              "best_overall_algorithm": str(score_df.iloc[0].algorithm),
              "best_overall_silhouette": round(float(score_df.iloc[0].silhouette), 4)}
    pd.DataFrame([result]).to_json(metrics_dir / "run_summary.json", orient="records", indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run customer segmentation experiments.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/Mall_Customers_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(run_clustering(args.data, args.output))


if __name__ == "__main__":
    main()
