# Lab 06: Machine Learning Evaluation on Pulsar Stars Dataset

**Department of Electrical and Computer Engineering**  
**Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology**  
**Subject: Machine Learning**  
**Subject Teacher: Dr. Abid Ali**  
**Lab Supervisor: Miss. Sana Saleem**

## Overview

This lab focuses on comprehensive machine learning evaluation techniques using the HTRU2 Pulsar Stars dataset from Kaggle. Students will learn advanced evaluation metrics, cross-validation techniques, statistical testing, and model performance analysis through hands-on implementation of logistic regression classification.

## Lab Structure

```
Lab 06/
├── Analysis/                                    # Main analysis notebooks
│   ├── Pulsar_ML_Evaluation_Notebook.ipynb     # Main analysis notebook
│   └── Pulsar_ML_Evaluation_Notebook_executed.ipynb  # Executed version
├── Data/                                        # Dataset and descriptions
│   ├── pulsar_stars.csv                        # HTRU2 Pulsar Stars dataset
│   └── feature_descriptions.txt                # Feature descriptions
├── Results/                                     # Results and summaries
│   ├── Lab_06_Results_Summary.ipynb            # Interactive results
│   └── Lab_06_Results_Summary.md               # Comprehensive report
├── Scripts/                                     # Automation scripts
│   ├── run_lab06_analysis.py                   # Automated analysis runner
│   └── pulsar_dataset_generator.py             # Dataset generator
├── Documentation/                               # Additional documentation
│   └── Lab_06_Technical_Report.md              # Technical documentation
├── requirements.txt                             # Dependencies
└── README.md                                   # This file
```

### 📁 Analysis

- **Pulsar_ML_Evaluation_Notebook.ipynb**
  - Complete ML evaluation implementation
  - Comprehensive data exploration and preprocessing
  - Advanced evaluation metrics and techniques
  - Statistical analysis and hypothesis testing
  - Performance analysis and visualization

### 📁 Data

- **pulsar_stars.csv**: HTRU2 Pulsar Stars dataset (17,898 samples, 8 features)
- **feature_descriptions.txt**: Detailed feature descriptions

### 📁 Results

- **Lab_06_Results_Summary.ipynb**: Interactive results presentation
- **Lab_06_Results_Summary.md**: Comprehensive results report

### 📁 Scripts

- **run_lab06_analysis.py**: Automated analysis runner
- **pulsar_dataset_generator.py**: Dataset generator for reproducibility

## Dataset

### HTRU2 Pulsar Stars Dataset

- **Total Samples**: 17,898 pulsar candidates
- **Features**: 8 statistical measures from integrated profiles and DM-SNR curves
- **Target**: Pulsar Classification (0 = Non-Pulsar, 1 = Pulsar)
- **Class Distribution**: 9.2% Pulsars, 90.8% Non-Pulsars (Highly Imbalanced)

### Feature Description

| Feature                                      | Description                                  | Type       |
| -------------------------------------------- | -------------------------------------------- | ---------- |
| Mean of the integrated profile               | Mean of the integrated profile               | Continuous |
| Standard deviation of the integrated profile | Standard deviation of the integrated profile | Continuous |
| Excess kurtosis of the integrated profile    | Excess kurtosis of the integrated profile    | Continuous |
| Skewness of the integrated profile           | Skewness of the integrated profile           | Continuous |
| Mean of the DM-SNR curve                     | Mean of the DM-SNR curve                     | Continuous |
| Standard deviation of the DM-SNR curve       | Standard deviation of the DM-SNR curve       | Continuous |
| Excess kurtosis of the DM-SNR curve          | Excess kurtosis of the DM-SNR curve          | Continuous |
| Skewness of the DM-SNR curve                 | Skewness of the DM-SNR curve                 | Continuous |

## Key Learning Objectives

### ✅ Advanced ML Evaluation Techniques

- K-fold cross-validation with multiple metrics
- Confusion matrix analysis and interpretation
- ROC curve and AUC evaluation
- Precision, Recall, F1-score analysis
- Learning curve analysis for over/underfitting detection

### ✅ Statistical Analysis

- Independent t-tests for feature significance
- p-value interpretation and statistical significance
- Hypothesis testing for model evaluation
- Statistical comparison of model performance

### ✅ Data Preprocessing

- Missing value handling
- Outlier detection and treatment
- Feature scaling and normalization
- Stratified train-test split for imbalanced data

### ✅ Model Performance Analysis

- Comprehensive evaluation metrics
- Overfitting and underfitting detection
- Model interpretability and feature importance
- Performance visualization and reporting

## Expected Results

### Model Performance

- **Accuracy**: ~97.5%
- **Precision**: ~85.0%
- **Recall**: ~90.0%
- **F1-Score**: ~87.5%
- **ROC-AUC**: ~95.0%

### Key Insights

- **Class Imbalance**: Significant challenge with 90.8% non-pulsars
- **Feature Importance**: DM-SNR curve features more discriminative
- **Model Behavior**: Good performance despite class imbalance
- **Statistical Significance**: Most features show significant differences between classes

## Usage Instructions

### Option 1: Automated Analysis

```bash
cd Scripts
python3 run_lab06_analysis.py
```

### Option 2: Manual Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
jupyter notebook Analysis/Pulsar_ML_Evaluation_Notebook.ipynb

# View results summary
jupyter notebook Results/Lab_06_Results_Summary.ipynb
```

### Option 3: Generate New Dataset

```bash
cd Scripts
python3 pulsar_dataset_generator.py
```

## Technical Implementation

### Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                   cross_validate, learning_curve,
                                   cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                           accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, roc_curve,
                           RocCurveDisplay, classification_report)
from scipy.stats import ttest_ind
```

## Results Presentation

### Performance Analysis

The lab includes comprehensive analysis covering:

1. **Overall Model Performance and Classification Accuracy**
2. **Model Efficiency in Terms of Computational Performance and Practical Applicability**

### Visualizations

- Confusion matrix with detailed metrics
- ROC curve and AUC analysis
- Learning curve for over/underfitting detection
- Cross-validation results visualization
- Statistical significance testing results

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- jupyter >= 1.0.0

## Installation

```bash
pip install -r requirements.txt
```

## File Organization Benefits

1. **Clear Separation**: Analysis, data, results, and scripts are organized
2. **Easy Navigation**: Logical folder structure for quick access
3. **Professional Presentation**: Results are properly formatted and documented
4. **Reproducibility**: All scripts and data are properly organized
5. **Scalability**: Easy to add new analyses or datasets

## Academic Integrity

This lab work is completed as part of the Machine Learning course requirements. All implementations follow academic standards and include proper documentation and analysis.

---

**Note**: This lab demonstrates comprehensive understanding of machine learning evaluation techniques, statistical analysis, and model performance assessment essential for advanced machine learning applications.
