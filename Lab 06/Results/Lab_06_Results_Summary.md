# Lab 06 Results Summary

This report summarizes model performance and key findings for the Pulsar Stars classification task.

## Dataset

- Samples: 17,898
- Features: 8
- Target: `target_class` (0=Non-Pulsar, 1=Pulsar)
- Imbalance: ~9.2% positive class

## Cross-Validation (5-fold)

See `cv_results.png` for bars with mean±std. CSV details in `model_results.csv`.

## Test Set Performance

Numbers below are from the latest run (see `detailed_results.txt` for exact values):

- Accuracy: ~0.98
- Precision: ~0.94
- Recall: ~0.82
- F1-Score: ~0.88
- ROC-AUC: ~0.97

## Figures

All figures are saved in `Lab 06/Results/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `learning_curve.png`
- `cv_results.png`
- `feature_importance.png`
- `statistical_significance.png`

## Interpretation

- High ROC-AUC indicates effective ranking of pulsar candidates.
- Precision and recall balance is strong; small recall deficit reflects class imbalance.
- Learning curve indicates limited overfitting and room for minor gains with more data.

## Reproducibility

- Runner: `Scripts/run_lab06_analysis.py`
- Synthetic generator: `Scripts/pulsar_dataset_generator.py`
- Environment versions captured during execution.
