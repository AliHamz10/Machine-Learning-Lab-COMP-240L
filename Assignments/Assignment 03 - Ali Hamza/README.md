# CCP: Comparative Machine Learning Framework for Heart Disease Prediction

**Course:** COMP-240L Machine Learning Lab  
**Student:** Ali Hamza  
**Dataset:** Cleveland Heart Disease Dataset  
**Weightage:** 5%  
**Submission Deadline:** 10 November 2025

## Project Overview

This project implements a comprehensive comparative machine learning framework to predict the presence or absence of heart disease using patient clinical data. The framework evaluates five different algorithms to determine which model offers the best trade-off between accuracy, precision, and interpretability for medical diagnostic prediction.

## Objectives

1. Develop and critically evaluate a predictive diagnostic system for heart disease
2. Implement multiple machine learning algorithms (Decision Tree, Random Forest, SVM, ANN, KNN)
3. Perform comprehensive data preprocessing and feature engineering
4. Conduct hyperparameter tuning for optimal model performance
5. Evaluate models using multiple metrics suitable for imbalanced medical data
6. Provide data-driven justification for model selection

## Dataset

**Cleveland Heart Disease Dataset**
- **Source:** UCI ML Repository
- **Features:** 13 clinical features (age, sex, chest pain type, etc.)
- **Target:** Binary classification (0 = no disease, 1 = disease)
- **Samples:** 303 instances
- **Characteristics:** Contains missing values, potential class imbalance

### Features:
1. age - Age in years
2. sex - Sex (1 = male; 0 = female)
3. cp - Chest pain type (0-3)
4. trestbps - Resting blood pressure
5. chol - Serum cholesterol in mg/dl
6. fbs - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. restecg - Resting electrocardiographic results
8. thalach - Maximum heart rate achieved
9. exang - Exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest
11. slope - Slope of the peak exercise ST segment
12. ca - Number of major vessels colored by flourosopy
13. thal - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

## Project Structure

```
Assignment 03 - Ali Hamza/
├── Scripts/
│   └── heart_disease_ml_framework.py    # Main Python implementation
├── Notebooks/
│   └── Heart_Disease_ML_Analysis.ipynb # Jupyter notebook version
├── Data/
│   ├── download_dataset.py             # Dataset download script
│   └── heart_disease.csv                # Dataset (downloaded)
├── Results/
│   ├── heart_disease_eda.png           # EDA visualizations
│   ├── roc_curves.png                  # ROC curves comparison
│   ├── confusion_matrices.png          # Confusion matrices
│   ├── model_comparison.png            # Model comparison charts
│   ├── feature_importance.png         # Feature importance plots
│   ├── model_comparison.csv            # Results comparison table
│   ├── summary_report.txt              # Text summary report
│   └── saved_models/                   # Trained models (pickle files)
├── Documentation/
│   └── CCP_Report_Template.md          # Report template
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Assignments/Assignment 03 - Ali Hamza"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset (optional - script will download automatically):**
   ```bash
   cd Data
   python download_dataset.py
   cd ..
   ```

## Usage

### Option 1: Run Python Script

```bash
cd Scripts
python heart_disease_ml_framework.py
```

The script will:
- Download the dataset (if not present locally)
- Perform exploratory data analysis
- Preprocess the data
- Train all 5 models
- Perform hyperparameter tuning for RF and SVM
- Evaluate and compare all models
- Generate all visualizations
- Save results to `../Results/` directory

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook Notebooks/Heart_Disease_ML_Analysis.ipynb
```

The notebook provides an interactive environment with step-by-step explanations.

## Models Implemented

1. **Decision Tree (DT)**
   - Interpretable tree-based model
   - Feature importance analysis available
   - No scaling required

2. **Random Forest (RF)**
   - Ensemble of decision trees
   - Hyperparameter tuning performed
   - Feature importance visualization

3. **Support Vector Machine (SVM)**
   - RBF kernel with probability estimates
   - Hyperparameter tuning performed
   - Requires feature scaling

4. **Artificial Neural Network (ANN)**
   - Multi-layer Perceptron (MLP)
   - Hidden layers: (100, 50)
   - Early stopping enabled
   - Requires feature scaling

5. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - k=5 neighbors
   - Requires feature scaling

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Minimizing false positives
- **Recall**: Minimizing false negatives
- **F1-Score**: Balanced precision-recall metric
- **AUC-ROC**: Discriminative ability
- **Cross-Validation**: 5-fold CV for robustness

## Key Features

- **Comprehensive EDA**: Statistical analysis and visualizations
- **Data Preprocessing**: Missing value handling, feature scaling, SMOTE for imbalance
- **Hyperparameter Tuning**: GridSearchCV for RF and SVM
- **Model Comparison**: Side-by-side evaluation of all models
- **Professional Visualizations**: Publication-quality charts and graphs
- **Well-Documented Code**: Clear comments and docstrings
- **Modular Design**: Easy to extend and modify

## Results

After running the framework, you'll find:

1. **Visualizations** in `Results/`:
   - EDA plots showing data distributions and correlations
   - ROC curves comparing all models
   - Confusion matrices for each model
   - Model comparison bar charts
   - Feature importance plots

2. **Data Files**:
   - `model_comparison.csv`: Detailed performance metrics
   - `summary_report.txt`: Text summary of results

3. **Trained Models**:
   - All models saved as pickle files in `Results/saved_models/`

## Best Model Selection

The framework automatically identifies the best model based on accuracy, but selection should consider:
- **Medical Context**: False negatives (missing disease) are critical
- **Interpretability**: Clinicians need to understand predictions
- **Performance**: Balance between accuracy, precision, and recall
- **Robustness**: Cross-validation performance

## Report

See `Documentation/CCP_Report_Template.md` for the structured report template covering:
- Abstract
- Introduction
- Methodology
- Results & Analysis
- Model Comparison & Discussion
- Conclusion & Recommendations

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- imbalanced-learn >= 0.10.0
- jupyter >= 1.0.0

## Notes

- The dataset is automatically downloaded from UCI ML Repository
- Missing values are handled using median imputation
- Class imbalance is addressed using SMOTE if ratio < 0.6
- All random operations use `random_state=42` for reproducibility
- Results are saved with timestamps for version tracking

## References

- UCI Machine Learning Repository: Heart Disease Dataset
- Scikit-learn Documentation
- Imbalanced-learn Documentation

## License

This project is for educational purposes as part of COMP-240L Machine Learning Lab course.

