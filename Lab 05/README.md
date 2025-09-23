# Lab 05: Logistic Regression Classification Analysis

**Department of Electrical and Computer Engineering**  
**Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology**  
**Subject: Machine Learning**  
**Subject Teacher: Dr. Abid Ali**  
**Lab Supervisor: Miss. Sana Saleem**

## Overview

This lab focuses on implementing logistic regression for binary classification using a heart disease prediction dataset. The lab demonstrates comprehensive data preprocessing, model training, evaluation, and performance analysis with detailed results presentation.

## Lab Structure

```
Lab 05/
├── Analysis/                                    # Main analysis notebooks
│   └── Lab_05_Logistic_Regression_Analysis.ipynb
├── Data/                                        # Dataset and descriptions
│   ├── heart_disease_dataset.csv
│   └── feature_descriptions.txt
├── Results/                                     # Results and summaries
│   ├── Lab_05_Results_Summary.ipynb
│   └── Lab_05_Results_Summary.md
├── Scripts/                                     # Automation scripts
│   ├── run_lab05_analysis.py
│   └── heart_disease_dataset.py
├── requirements.txt                             # Dependencies
└── README.md                                   # This file
```

### 📁 Analysis
- **Lab_05_Logistic_Regression_Analysis.ipynb**
  - Complete logistic regression implementation
  - Comprehensive data exploration and preprocessing
  - Model evaluation with multiple metrics
  - Performance analysis and visualization
  - Two detailed performance analysis paragraphs

### 📁 Data
- **heart_disease_dataset.csv**: Main dataset (1000 samples, 13 features)
- **feature_descriptions.txt**: Detailed feature descriptions

### 📁 Results
- **Lab_05_Results_Summary.ipynb**: Interactive results presentation
- **Lab_05_Results_Summary.md**: Comprehensive results report

### 📁 Scripts
- **run_lab05_analysis.py**: Automated analysis runner
- **heart_disease_dataset.py**: Dataset generator

## Dataset

### Heart Disease Prediction Dataset
- **Total Samples**: 1,000 patients
- **Features**: 13 medical indicators
- **Target**: Heart Disease (0 = No Disease, 1 = Disease)
- **Class Distribution**: 78.3% Heart Disease, 21.7% No Heart Disease

### Feature Description
| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Continuous |
| sex | Sex (0 = female, 1 = male) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mmHg) | Continuous |
| chol | Serum cholesterol (mg/dl) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting electrocardiographic results | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Continuous |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Categorical |
| thal | Thalassemia (0-2) | Categorical |

## Key Learning Objectives

### ✅ Data Preprocessing
- Missing value handling
- Outlier detection and treatment
- Categorical variable encoding
- Feature scaling and normalization
- Train-test split

### ✅ Logistic Regression Implementation
- Binary classification model
- Feature importance analysis
- Model interpretation
- Probability threshold optimization

### ✅ Model Evaluation
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix analysis
- ROC curve and AUC
- Precision-Recall curve
- Cross-validation

### ✅ Data Visualization
- Distribution analysis
- Correlation heatmaps
- Feature importance charts
- Model performance visualizations
- ROC and Precision-Recall curves

## Expected Results

### Model Performance
- **Accuracy**: ~87.5%
- **Precision**: ~89.0%
- **Recall**: ~91.0%
- **F1-Score**: ~90.0%
- **ROC-AUC**: ~92.3%

### Key Features
- **Most Important**: ST depression, vessel count, thalassemia, chest pain
- **Protective**: Higher maximum heart rate during exercise
- **Clinical Relevance**: All features align with medical knowledge

## Usage Instructions

### Option 1: Automated Analysis
```bash
cd Scripts
python3 run_lab05_analysis.py
```

### Option 2: Manual Analysis
```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
jupyter notebook Analysis/Lab_05_Logistic_Regression_Analysis.ipynb

# View results summary
jupyter notebook Results/Lab_05_Results_Summary.ipynb
```

### Option 3: Generate New Dataset
```bash
cd Scripts
python3 heart_disease_dataset.py
```

## Technical Implementation

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, classification_report)
```

## Results Presentation

### Performance Efficiency Analysis
The lab includes two detailed paragraphs analyzing:
1. **Overall Model Performance and Classification Accuracy**
2. **Model Efficiency in Terms of Computational Performance and Practical Applicability**

### Visualizations
- Comprehensive 9-panel visualization dashboard
- Confusion matrix with detailed metrics
- ROC and Precision-Recall curves
- Feature importance analysis
- Cross-validation results
- Performance metrics comparison

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

**Note**: This lab demonstrates comprehensive understanding of logistic regression, binary classification, and model evaluation techniques essential for machine learning applications.
