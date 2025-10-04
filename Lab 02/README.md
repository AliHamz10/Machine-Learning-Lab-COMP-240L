# Lab 02: Linear Regression Analysis on Insurance Dataset

**Department Of Electrical & Computer Engineering**  
**Machine Learning Lab - COMP 240L**  
**Lab Coordinator: Ms. Sana Saleem**  
**Course Instructor: Dr. Abid Ali**  
**Program: BS(AI) -F23, Semester: 5th**  
**Deadline: 13th September 2025**

## Overview

This lab implements a comprehensive linear regression analysis on an insurance dataset to predict medical charges based on various demographic and health factors.

## Dataset

- **File**: `insurance.csv`
- **Records**: 1,338 insurance claims
- **Features**: age, gender, bmi, children, smoker, region
- **Target**: charges (medical insurance charges)

## Files Structure

```
Lab 02/
├── Notebooks/                                  # Jupyter notebooks
│   └── Lab_02_Linear_Regression_Analysis.ipynb
├── Data/                                       # Dataset files
│   └── insurance.csv
├── Results/                                    # Analysis results
│   └── lab02_comprehensive_analysis.png
├── Scripts/                                    # Python scripts
│   └── run_lab02_analysis.py
├── Environment/                                # Virtual environment
│   ├── bin/                                    # Python executables
│   ├── lib/                                    # Installed packages
│   └── pyvenv.cfg                              # Environment config
├── requirements.txt                            # Python dependencies
└── README.md                                   # This documentation
```

## Quick Start

### Option 1: Run Python Script

```bash
cd "Lab 02"
python Scripts/run_lab02_analysis.py
```

### Option 2: Use Jupyter Notebook

```bash
cd "Lab 02"
jupyter notebook Notebooks/Lab_02_Linear_Regression_Analysis.ipynb
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Analysis Components

### 1. Data Exploration

- Dataset overview and statistics
- Missing value analysis
- Distribution analysis of all variables
- Correlation analysis

### 2. Data Preprocessing

- Categorical variable encoding
- Feature scaling
- Train-test split (80-20)

### 3. Model Implementation

- Linear regression model training
- Feature importance analysis
- Model interpretation

### 4. Model Evaluation

- Training and test set metrics
- R², RMSE, MAE calculations
- Overfitting analysis

### 5. Visualizations

- Distribution plots
- Scatter plots (age vs charges, BMI vs charges)
- Actual vs Predicted plots
- Residual plots
- Feature importance visualization
- Comprehensive analysis dashboard

### 6. Results and Insights

- Model performance summary
- Business insights
- Feature impact analysis
- Sample predictions
- Conclusions and recommendations

## Expected Results

- **R² Score**: Typically 0.75-0.85 (75-85% variance explained)
- **Key Findings**: Smoking status has the strongest impact on charges
- **Model Performance**: Good generalization with reasonable prediction accuracy

## Key Insights

1. **Smoking Status**: Most significant predictor of medical charges
2. **Age and BMI**: Strong positive correlation with charges
3. **Gender and Region**: Moderate impact on charges
4. **Children**: Minimal impact on charges

## Model Limitations

- Assumes linear relationships between features and target
- May not capture complex feature interactions
- Could benefit from non-linear models for better accuracy

## Recommendations

- Consider polynomial features for non-linear relationships
- Try ensemble methods (Random Forest, XGBoost)
- Feature engineering for improved performance
- Collect more data for better model accuracy

## Output Files

- `lab02_comprehensive_analysis.png` - Comprehensive visualization dashboard
- All plots and analysis results displayed in notebook

## Contact

For questions about this lab, contact:

- Lab Coordinator: Ms. Sana Saleem
- Course Instructor: Dr. Abid Ali
