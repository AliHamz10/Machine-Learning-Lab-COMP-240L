# Lab 03: Linear Regression Model Implementation and Analysis

**Department of Electrical and Computer Engineering**  
**Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology**  
**Subject: Machine Learning**  
**Subject Teacher: Dr. Abid Ali**  
**Lab Supervisor: Miss. Sana Saleem**

## Overview

This lab focuses on implementing linear regression models using real-world datasets, exploring data preprocessing techniques, and evaluating model performance using various metrics. The lab includes both theoretical understanding and practical implementation of gradient descent algorithms.

## Lab Structure

### 📁 Practices
Contains individual practice notebooks demonstrating specific concepts:

1. **Practice_1_Single_Variable_Linear_Regression.ipynb**
   - Single variable linear regression implementation
   - Diabetes dataset analysis using Glucose as predictor
   - Outlier detection and removal using Z-score
   - Model evaluation and visualization

2. **Practice_2_Multi_Variable_Linear_Regression.ipynb**
   - Multi-variable linear regression with feature scaling
   - Correlation analysis and feature importance
   - StandardScaler implementation
   - Comprehensive model evaluation

3. **Practice_3_Classification_Metrics.ipynb**
   - Classification metrics evaluation (Accuracy, Precision, Recall, F1-Score)
   - Binary classification using threshold conversion
   - Confusion matrix analysis
   - ROC and Precision-Recall curves

4. **Practice_4_Cost_Function_Gradient_Descent.ipynb**
   - Custom gradient descent implementation from scratch
   - Cost function visualization over iterations
   - Comparison with sklearn implementation
   - Learning curve analysis

### 📁 Lab Tasks
Contains the main lab assignment:

- **Lab_03_Insurance_Dataset_Analysis.ipynb**
  - Complete insurance dataset analysis
  - All practices integrated into comprehensive analysis
  - Real-world application of linear regression
  - Advanced visualization and evaluation

## Datasets

### Diabetes Dataset
- **File**: `diabetes.csv`
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0 or 1)
- **Purpose**: Practice implementations and metric evaluation

### Insurance Dataset
- **File**: `Insurance.csv`
- **Features**: age, gender, bmi, children, smoker, region
- **Target**: charges (medical insurance costs)
- **Purpose**: Main lab task with real-world application

## Key Learning Objectives

### ✅ Data Preprocessing
- Missing value handling
- Outlier detection using Z-score
- Categorical variable encoding (LabelEncoder)
- Feature scaling (StandardScaler)

### ✅ Linear Regression Implementation
- Single variable regression
- Multi-variable regression
- Feature importance analysis
- Model interpretation

### ✅ Gradient Descent Algorithm
- Custom implementation from scratch
- Cost function optimization
- Learning rate tuning
- Convergence analysis

### ✅ Model Evaluation
- Regression metrics (MSE, RMSE, MAE, R²)
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix analysis
- ROC and Precision-Recall curves

### ✅ Data Visualization
- Distribution analysis
- Correlation heatmaps
- Residual plots
- Feature importance charts
- Model performance comparisons

## Technical Implementation

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from scipy import stats
```

### Key Algorithms

#### Gradient Descent Implementation
```python
def gradient_descent(X, y, theta, bias, learning_rate, epochs):
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, theta) + bias
        d_theta = (1 / m) * np.dot(X.T, (predictions - y))
        d_bias = (1 / m) * np.sum(predictions - y)
        
        theta -= learning_rate * d_theta
        bias -= learning_rate * d_bias
        
        cost = compute_cost(X, y, theta, bias)
        cost_history.append(cost)
    
    return theta, bias, cost_history
```

## Results and Insights

### Insurance Dataset Analysis
- **Dataset Size**: 1,338 samples
- **Features**: 6 (age, gender, bmi, children, smoker, region)
- **Model Performance**: R² ≈ 0.75-0.80
- **Key Findings**:
  - Smoking status is the most important predictor
  - Age and BMI significantly impact insurance costs
  - Gender and region show moderate influence
  - Number of children has minimal impact

### Model Comparison
- **Sklearn vs Gradient Descent**: Similar performance validates custom implementation
- **Regression Metrics**: MSE, R², MAE show consistent results
- **Classification Metrics**: Good accuracy for binary classification tasks

## File Structure
```
Lab 03/
├── Practices/                                    # Individual practice notebooks
│   ├── Practice_1_Single_Variable_Linear_Regression.ipynb
│   ├── Practice_2_Multi_Variable_Linear_Regression.ipynb
│   ├── Practice_3_Classification_Metrics.ipynb
│   └── Practice_4_Cost_Function_Gradient_Descent.ipynb
├── Lab Tasks/                                    # Main lab assignment
│   ├── Lab_03_Insurance_Dataset_Analysis.ipynb
│   └── Insurance.csv
├── diabetes.csv                                  # Diabetes dataset
├── insurance[1].csv                             # Additional insurance data
├── yahoo_data[1].xlsx                           # Financial data
├── Lab 03 - Linear Regression, Cost Function and Gradient Descent.pdf
├── Python Libraries.pdf                         # Library documentation
├── requirements.txt                             # Python dependencies
├── run_lab03_analysis.py                        # Analysis script
└── README.md                                    # This documentation
```

## Usage Instructions

1. **Run Individual Practices**: Execute each practice notebook to understand specific concepts
2. **Complete Lab Tasks**: Run the main insurance analysis notebook
3. **Experiment with Parameters**: Modify learning rates, epochs, and thresholds
4. **Analyze Results**: Review visualizations and metrics for insights

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Academic Integrity

This lab work is completed as part of the Machine Learning course requirements. All implementations follow academic standards and include proper documentation and analysis.

---

**Note**: This lab demonstrates comprehensive understanding of linear regression, gradient descent optimization, and model evaluation techniques essential for machine learning applications.
