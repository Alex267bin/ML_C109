# Dataset Documentation – Mall Customer Segmentation

## 1. Dataset overview

This practice uses the **Mall Customer Segmentation** dataset supplied as `Mall_Customers.csv`.

- Number of records: **200 customers**
- Number of columns: **5**
- File format: CSV
- Data type: tabular customer-demographic and spending data
- Intended task: unsupervised customer segmentation

> **Source/provenance note:** The CSV was supplied to the project workspace by the student. Its structure matches the commonly circulated Mall Customer Segmentation dataset. However, the original Kaggle page, dataset owner, version, and license were not included with the uploaded file, so this document does not claim a verified Kaggle URL or license. Add the exact Kaggle dataset URL and license here if available.

## 2. Data dictionary

| Column | Meaning | Type | Use in this practice |
|---|---|---|---|
| `CustomerID` | Unique customer identifier | Integer | Identification only; excluded from clustering |
| `Genre` | Customer gender category (Male/Female) | Categorical | Optional descriptive analysis; excluded from distance calculation |
| `Age` | Customer age in years | Integer | Clustering feature |
| `Annual_Income_(k$)` | Annual income in thousand US dollars | Integer | Clustering feature |
| `Spending_Score` | Spending score on a scale from 1 to 100 | Integer | Clustering feature |

The three clustering features are Age, Annual Income, and Spending Score. `CustomerID` is an identifier and must not be treated as a meaningful numeric feature.

## 3. Dataset validation

The supplied file was inspected before being documented.

| Check | Result |
|---|---:|
| Rows | 200 |
| Columns | 5 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate CustomerID values | 0 |
| Age range | 18–70 |
| Annual income range | 15–137 (k$) |
| Spending score range | 1–99 |

Descriptive statistics for the three numeric clustering features:

| Feature | Mean | Standard deviation | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| Age | 38.85 | 13.97 | 18 | 36 | 70 |
| Annual Income (k$) | 60.56 | 26.26 | 15 | 61.5 | 137 |
| Spending Score | 50.20 | 25.82 | 1 | 50 | 99 |

These values describe the supplied CSV file. They are not model performance metrics.

## 4. Preprocessing and feature selection

1. Load the CSV with pandas.
2. Normalize column names in the processing script where necessary (for example, `Genre` may be mapped to `Gender` internally).
3. Check missing values and inspect outliers using the IQR method.
4. Select `Age`, `Annual_Income_(k$)`, and `Spending_Score` as the feature matrix.
5. Apply `StandardScaler` before distance-based clustering.
6. Keep `CustomerID` and `Genre` out of the clustering feature matrix. They may be used for record identification and post-clustering descriptions only.

## 5. How to use the file

Place the dataset at:

```text
data/raw/Mall_Customers.csv
```

Run the customer clustering pipeline from the Practice 3 directory:

```bash
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
```

## 6. Limitations

- This dataset contains only 200 customer records; results should not be generalized to all mall customers.
- The dataset documentation supplied here does not establish how or when the data was collected.
- The meaning and construction of the Spending Score are not further defined by the CSV itself.
- The uploaded file alone does not provide a verified Kaggle source URL or reuse license. Cite the original Kaggle dataset page and follow its license if the course requires formal source attribution.
