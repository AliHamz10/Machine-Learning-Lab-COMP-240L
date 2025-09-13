# Machine Learning Lab - COMP 240L

**Course:** BS(AI) - F23 | **Semester:** 5th  
**Lab Coordinator:** Ms. Sana Saleem  
**Course Instructor:** Dr. Abid Ali

This repository contains the lab work and projects for COMP 240L Machine Learning course.

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

## Technical Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Environment:** Jupyter Notebook
- **Data Processing:** Missing value imputation, categorical encoding
- **Machine Learning:** Linear Regression, train-test split
- **Visualization:** Matplotlib, Seaborn

## Repository Structure

```
Machine-Learning-Lab-COMP-240L/
├── Lab 01/                          # Completed
│   ├── Lab 01.ipynb                 # Main assignment notebook
│   ├── Lab 01_executed.ipynb        # Executed version with outputs
│   └── ...
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

2. **Set up environment:**

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

## Author

**Ali Hamza**  
Registration Number: **B23F0063AI106**
BS(AI) - F23, Semester 5th  
Machine Learning Lab - COMP 240L

**Zarmeena Jawad**  
Registration Number: **B23F0115AI125**
BS(AI) - F23, Semester 5th  
Machine Learning Lab - COMP 240L
