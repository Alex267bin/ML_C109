# Practice 2 – Predicting Product Sales: Report

> Complete this report only after running the pipeline. Do not invent metrics or observations.

## 1. Problem definition
Describe the retail sales prediction problem, prediction target (`sales`) and prediction grain (date, store and product family).

## 2. Dataset description
List each CSV used, row/column counts, date range, target variable and available predictors. Explain the missing assignment concepts documented in DATASET_NOTES.md.

## 3. Exploratory data analysis
Insert the generated figures from `outputs/eda/`. Discuss target distribution, monthly trend, product-family sales, missing values and duplicate records based on actual output.

## 4. Data preprocessing
Explain date parsing, joins, missing-value handling, categorical encoding and chronological validation split.

## 5. Feature engineering
Explain calendar features, promotion count, store attributes, oil price and holiday indicator. Discuss why unavailable customer/review/marketing features were not fabricated.

## 6. Models
Describe Linear Regression, Decision Tree Regressor and Random Forest Regressor, including the chosen hyperparameters.

## 7. Evaluation
Insert actual MAE, MSE and RMSE from `outputs/metrics/model_comparison.csv`. Identify the best model by validation RMSE.

## 8. Discussion and limitations
Discuss time-series validation, potential distribution shift, lack of customer-level data and the fact that the pipeline is a local experiment rather than a production deployment.

## 9. Conclusion and future work
Summarize measured findings and propose leakage-safe lag features, recursive/multi-step forecasting, hyperparameter search and model monitoring.
