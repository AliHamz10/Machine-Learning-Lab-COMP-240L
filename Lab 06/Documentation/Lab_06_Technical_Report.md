# Lab 06 Technical Report: ML Evaluation on Pulsar Stars

## Objective

Evaluate a logistic regression classifier on the HTRU2 Pulsar Stars dataset with rigorous metrics and diagnostics: stratified k-fold CV, confusion matrix, ROC-AUC, learning curves, and feature-wise statistical testing.

## Dataset

- Samples: 17,898
- Features: 8 statistical descriptors from integrated profiles and DM-SNR curves
- Target: `target_class` (0 = Non-Pulsar, 1 = Pulsar)
- Imbalance: ~9.2% positive class

## Methodology

1. Preprocessing: NaN/inf handling, numeric coercion, standardization (via `StandardScaler`).
2. Model: `LogisticRegression` (`lbfgs`, max_iter=1000) within a `Pipeline`.
3. Validation: Stratified 5-fold cross-validation with metrics: accuracy, precision, recall, f1, roc_auc.
4. Hold-out test set (25%) for final evaluation.
5. Diagnostics and Visualization:
   - Confusion Matrix
   - ROC Curve with AUC
   - Learning Curve (bias/variance assessment)
   - Cross-validation bar chart with mean±std
   - Coefficient-based feature importance
6. Statistical Analysis: Welch’s t-test per feature (class 0 vs 1) with p<0.05 significance threshold.

## Results (Summary)

See `Results/model_results.csv` and `Results/detailed_results.txt`.

- Typical performance (will vary slightly by split):
  - Accuracy ≈ 0.98
  - Precision ≈ 0.94
  - Recall ≈ 0.82
  - F1 ≈ 0.88
  - ROC-AUC ≈ 0.97
- All eight features exhibit statistically significant mean differences between classes (p < 0.05).

## Interpretation

- High ROC-AUC and strong precision/recall indicate effective separation despite imbalance.
- Slight recall deficit reflects the inherent challenge of detecting rare positives; threshold tuning or class weighting could increase recall.
- Learning curve shows minimal overfitting; additional data may yield small gains.

## Recommendations

- Consider class weighting or threshold calibration to trade precision for higher recall depending on application.
- Explore non-linear models (e.g., Gradient Boosting) and PR-AUC for deeper imbalance analysis.
- Perform calibration curves to verify probability quality for downstream decision-making.

## Reproducibility

- Automated runner: `Scripts/run_lab06_analysis.py`
- Synthetic data generator: `Scripts/pulsar_dataset_generator.py`
- Figures and tables saved under `Results/` for inspection and reporting.
