# Machine Learning Lab - COMP 240L

> Hands-on ML labs covering regression, optimization, metrics, and CNN transfer learning.

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/Python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-Academic-lightgrey)

**Course:** BS(AI) - F23 | **Semester:** 5th  
**Lab Coordinator:** Ms. Sana Saleem  
**Course Instructor:** Dr. Abid Ali

This repository contains the lab work and projects for COMP 240L Machine Learning course.

## Table of Contents

- [Lab Progress](#lab-progress)
- [Technical Stack](#technical-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [How to Run Each Lab](#how-to-run-each-lab)
- [Lab 01 Results Summary](#lab-01-results-summary)
- [Assignment Details](#assignment-details)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## Lab Progress

### Lab 01 - Machine Learning Assignment 01 (COMPLETED)

**Dataset:** Titanic Survival Prediction  
**Deadline:** 12th September 2025

**Overview:**
Lab 01 focused on implementing a complete machine learning pipeline using the Titanic dataset, which contains passenger information and survival outcomes. The primary objective was to predict passenger survival using linear regression while handling real-world data challenges including missing values, categorical variables, and feature engineering. This lab provided hands-on experience with data preprocessing, exploratory data analysis, model training, and evaluation techniques essential for machine learning projects.

**What We Accomplished:**

- Selected and loaded a dataset with missing values (Titanic dataset with 891 passengers)
- Performed comprehensive data exploration and statistical analysis
- Implemented data preprocessing techniques including missing value imputation
- Conducted correlation analysis to identify key predictive features
- Created multiple data visualizations for better understanding of the dataset
- Split the dataset into training (80%) and testing (20%) sets
- Trained a linear regression model for survival prediction
- Evaluated model performance using multiple metrics

**Key Results:**

- **Model Accuracy:** 80.45% on test set
- **R² Score:** 0.373 (explains 37.3% of variance)
- **Strongest Predictors:** Sex, Pclass, Fare
- **Missing Values:** All properly handled

**Files:**

- `Lab 01/Lab 01.ipynb` - Main assignment notebook
- `Lab 01/Lab 01_executed.ipynb` - Executed version with outputs
- `Lab Manuals/Lab 01.pdf` - Lab manual reference

### Lab 02 - Linear Regression on Insurance Dataset (COMPLETED)

**Dataset:** `Insurance.csv` (features: age, bmi, children, smoker, region; target: charges)

**Overview:**
Lab 02 performs an end-to-end regression analysis to predict insurance charges. It covers cleaning, EDA, encoding categorical variables, feature scaling, train/val/test split, and multiple regression models with metrics comparison. Visualizations highlight relationships (e.g., smoker vs charges, BMI trends) and residual diagnostics.

**What We Accomplished:**

- Loaded and validated the Insurance dataset
- Handled categorical variables (One-Hot Encoding), scaled numeric features
- Explored correlations and feature importance
- Trained Linear Regression, Lasso, Ridge; tuned hyperparameters
- Evaluated with RMSE/MAE/R² and residual plots
- Saved figures and provided a runnable script for reproducibility

**Key Results:**

- Strong signal from `smoker`, `bmi`, and `age`
- Best model reached low error with good generalization (see notebook for exact scores)

**Files:**

- `Lab 02/Lab_02_Linear_Regression_Analysis.ipynb` – Main notebook
- `Lab 02/Insurance.csv` – Dataset
- `Lab 02/run_lab02_analysis.py` – Runnable script
- `Lab 02/requirements.txt` – Lab-specific dependencies
- `Lab 02/README.md` – Notes and instructions

### Lab 03 - Linear Regression, Cost Function and Gradient Descent (COMPLETED)

**Datasets/Materials:** `Lab 03 - Linear Regression, Cost Function and Gradient Descent.pdf`, `Lab 03/Lab Tasks/Insurance.csv`, `diabetes.csv`

**Overview:**
Lab 03 focuses on core ML math and implementations: single- and multi-variable linear regression from scratch, cost function derivation, and gradient descent implementation/visualization. It also covers basic classification metrics.

**What We Accomplished:**

- Implemented univariate linear regression and visualized best-fit line
- Derived MSE cost function; implemented batch gradient descent
- Extended to multivariate regression with feature normalization
- Explored classification metrics (precision, recall, F1, ROC) in practice notebooks
- Organized tasks vs practices for clarity and repetition

**Key Results:**

- Convergent gradient descent with learning-rate tuning
- Clear visualization of cost landscapes and convergence curves

**Files:**

- `Lab 03/Practices/Practice_1_Single_Variable_Linear_Regression.ipynb`
- `Lab 03/Practices/Practice_2_Multi_Variable_Linear_Regression.ipynb`
- `Lab 03/Practices/Practice_3_Classification_Metrics.ipynb`
- `Lab 03/Practices/Practice_4_Cost_Function_Gradient_Descent.ipynb`
- `Lab 03/Lab Tasks/Lab_03_Insurance_Dataset_Analysis.ipynb`
- `Lab 03/run_lab03_analysis.py`, `Lab 03/requirements.txt`

### Lab 04 - Cat vs Tiger Image Classifier (COMPLETED)

**Dataset:** Custom image folders `data/cat`, `data/tiger` (+ optional `data/validation/...`) managed by scripts

**Overview:**
Lab 04 builds a high-accuracy image classifier using transfer learning (MobileNetV2), strong augmentation, training callbacks, and fine-tuning. Includes evaluation plots, confusion matrix, and a simple prediction helper. Data tooling scripts download and organize sample images.

**What We Accomplished:**

- Created dataset bootstrap scripts (`download_dataset.py`, `data_collection.py`)
- Implemented robust training pipeline with augmentation and fine-tuning
- Added evaluation: learning curves, classification report, confusion matrix
- Provided `predict_image()` for quick inference

**Key Results:**

- Working classifier with small sample data; accuracy improves with more images per class
- Reproducible environment via pinned `requirements.txt`

**Files:**

- `Lab 04/CatTigerClassifier/CatTigerClassifier_Improved.ipynb` – Main notebook
- `Lab 04/CatTigerClassifier/download_dataset.py` – Sample data bootstrap
- `Lab 04/CatTigerClassifier/data_collection.py` – Data helper
- `Lab 04/CatTigerClassifier/requirements.txt` – Pinned deps
- `Lab 04/CatTigerClassifier/.gitignore` – Ignore data/venv/models artifacts

## Technical Stack

- **Language:** Python 3.x
- **Core Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Deep Learning (Lab 04):** TensorFlow/Keras (MobileNetV2 transfer learning)
- **Environment:** Jupyter Notebook, virtualenv/venv
- **Data Processing:** Missing value imputation, categorical encoding, scaling
- **ML Topics:** Linear Regression, Gradient Descent, metrics, model selection, transfer learning
- **Visualization:** Matplotlib, Seaborn

## Repository Structure

```
Machine-Learning-Lab-COMP-240L/
├── Lab 01/                          # Completed
│   ├── Lab 01.ipynb                 # Main assignment notebook
│   ├── Lab 01_executed.ipynb        # Executed version with outputs
│   └── ...
├── Lab 02/                          # Completed
│   ├── Lab_02_Linear_Regression_Analysis.ipynb
│   ├── Insurance.csv
│   ├── run_lab02_analysis.py
│   ├── requirements.txt
│   └── README.md
├── Lab 03/                          # Completed
│   ├── Lab Tasks/
│   │   └── Lab_03_Insurance_Dataset_Analysis.ipynb
│   ├── Practices/
│   │   ├── Practice_1_Single_Variable_Linear_Regression.ipynb
│   │   ├── Practice_2_Multi_Variable_Linear_Regression.ipynb
│   │   ├── Practice_3_Classification_Metrics.ipynb
│   │   └── Practice_4_Cost_Function_Gradient_Descent.ipynb
│   ├── run_lab03_analysis.py
│   └── requirements.txt
├── Lab 04/                          # Completed
│   └── CatTigerClassifier/
│       ├── CatTigerClassifier_Improved.ipynb
│       ├── download_dataset.py
│       ├── data_collection.py
│       ├── requirements.txt
│       └── .gitignore
├── Lab Manuals/                     # Lab manuals and references
│   └── Lab 01.pdf
├── Lab Reports/                     # Lab reports (if any)
└── README.md                        # This file
```

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AliHamz10/Machine-Learning-Lab-COMP-240L.git
   cd Machine-Learning-Lab-COMP-240L
   ```

2. **Set up environment (global for classic ML labs):**

   ```bash
   # Create virtual environment
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

   # Install required packages
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```

For Lab 04 (TensorFlow), use lab-local requirements (see below).

## How to Run Each Lab

### Lab 01

Open and run the notebook:

```bash
cd Lab\ 01
jupyter notebook "Lab 01.ipynb"
```

### Lab 02

Run the analysis script or open the notebook:

```bash
cd Lab\ 02
python run_lab02_analysis.py  # or open Lab_02_Linear_Regression_Analysis.ipynb
```

### Lab 03

Open practice notebooks or the task notebook:

```bash
cd Lab\ 03
jupyter notebook Practices/Practice_1_Single_Variable_Linear_Regression.ipynb
```

### Lab 04 (Cat vs Tiger Classifier)

Use the lab-specific environment for reproducibility:

```bash
cd "Lab 04/CatTigerClassifier"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Optional: bootstrap a tiny sample dataset
python download_dataset.py

# Train and evaluate (open the notebook)
jupyter notebook CatTigerClassifier_Improved.ipynb
```

Data layout for Lab 04 (git-ignored):

```
data/
  cat/      # training cats
  tiger/    # training tigers
  validation/
    cat/    # validation cats
    tiger/  # validation tigers
```

## Lab 01 Results Summary

The Titanic Survival Prediction model achieved excellent performance:

- Successfully handled missing values in Age (177), Cabin (687), and Embarked (2) columns
- Identified key survival factors: Gender, Passenger Class, and Fare
- Achieved 80.45% accuracy with good generalization
- Comprehensive analysis with correlation matrices and visualizations

## Assignment Details

**Lab 01 Requirements Met:**

1. Dataset with missing values selected
2. Dataset read using pandas library
3. Preprocessing steps performed
4. Missing values handled appropriately
5. Correlation analysis completed
6. Data visualization via graphs
7. Dataset split (80% train, 20% test)
8. Linear regression model trained
9. Model evaluated with accuracy metrics

## Troubleshooting

- If Jupyter can’t find the kernel, ensure your venv is activated and run:
  ```bash
  python -m ipykernel install --user --name=ml_env --display-name "Python (ml_env)"
  ```
- On macOS with Homebrew Python (PEP 668), prefer a virtualenv and install inside it.
- For Lab 04 GPU/TF issues, start with CPU TensorFlow (already specified) and keep image sizes modest (e.g., 224×224).

## Author

**Ali Hamza**  
Registration Number: **B23F0063AI106**
BS(AI) - F23 Red, Semester 5th  
Machine Learning Lab - COMP 240L

**Zarmeena Jawad**  
Registration Number: **B23F0115AI125**
BS(AI) - F23 Red, Semester 5th  
Machine Learning Lab - COMP 240L
