# Lab 01: Introduction to Machine Learning

**Course:** Machine Learning Lab (COMP-240L)  
**Class:** BSAI F23 Red  
**Lab Number:** 01  
**Topic:** Introduction to Machine Learning Concepts and Python Libraries

## 📚 Lab Overview

This lab introduces fundamental machine learning concepts and provides hands-on experience with essential Python libraries for data science and machine learning. Students will learn the basics of data manipulation, visualization, and simple machine learning algorithms.

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Understand basic machine learning concepts and terminology
- Set up Python environment for machine learning
- Use NumPy for numerical computations
- Use Pandas for data manipulation and analysis
- Use Matplotlib for data visualization
- Use Scikit-learn for basic machine learning tasks
- Implement simple classification and regression algorithms

## 📋 Lab Content

### 1. Machine Learning Fundamentals
- **Supervised Learning:** Classification and Regression
- **Unsupervised Learning:** Clustering and Dimensionality Reduction
- **Model Training and Evaluation:** Train-test split, cross-validation
- **Overfitting and Underfitting:** Model complexity and generalization

### 2. Python Libraries Introduction
- **NumPy:** Numerical computing and array operations
- **Pandas:** Data manipulation and analysis
- **Matplotlib:** Data visualization and plotting
- **Scikit-learn:** Machine learning algorithms and tools

### 3. Hands-on Exercises
- Data loading and exploration
- Basic statistical analysis
- Data visualization techniques
- Simple model implementation
- Model evaluation and interpretation

## 📁 Files Structure

```
Lab 01/
├── Notebooks/                      # Jupyter notebooks
│   ├── Lab 01.ipynb               # Original lab notebook
│   └── Lab 01_executed.ipynb      # Executed lab notebook with results
├── Data/                          # Dataset files (if any)
├── Results/                       # Analysis results and outputs
└── README.md                      # This documentation
```

## 🛠️ Prerequisites

- Basic Python programming knowledge
- Understanding of statistics and probability
- Familiarity with data structures (lists, dictionaries, arrays)

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv lab01_env

# Activate virtual environment
# On Windows:
lab01_env\Scripts\activate
# On macOS/Linux:
source lab01_env/bin/activate

# Install required packages
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 2. Running the Lab
```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to Notebooks folder
# Open Lab 01.ipynb
# Execute cells sequentially
```

## 📊 Lab Exercises

### Exercise 1: NumPy Basics
- Array creation and manipulation
- Mathematical operations
- Statistical functions
- Array indexing and slicing

### Exercise 2: Pandas DataFrames
- Data loading from CSV files
- Data exploration and summary statistics
- Data cleaning and preprocessing
- Data filtering and selection

### Exercise 3: Data Visualization
- Line plots and scatter plots
- Histograms and bar charts
- Box plots and violin plots
- Customizing plots and styling

### Exercise 4: Scikit-learn Introduction
- Loading built-in datasets
- Train-test split
- Simple classification with Logistic Regression
- Model evaluation metrics

## 🎓 Key Concepts Covered

### Machine Learning Pipeline
1. **Data Collection:** Gathering relevant data
2. **Data Preprocessing:** Cleaning and preparing data
3. **Feature Engineering:** Creating meaningful features
4. **Model Selection:** Choosing appropriate algorithms
5. **Model Training:** Fitting models to data
6. **Model Evaluation:** Assessing model performance
7. **Model Deployment:** Using models for predictions

### Python Libraries Usage
- **NumPy:** `np.array()`, `np.mean()`, `np.std()`, `np.random`
- **Pandas:** `pd.read_csv()`, `df.head()`, `df.describe()`, `df.groupby()`
- **Matplotlib:** `plt.plot()`, `plt.hist()`, `plt.scatter()`, `plt.show()`
- **Scikit-learn:** `train_test_split()`, `LogisticRegression()`, `accuracy_score()`

## 📈 Expected Outcomes

After completing this lab, students should be able to:

- Load and explore datasets using Pandas
- Perform basic statistical analysis
- Create meaningful visualizations
- Implement simple machine learning models
- Evaluate model performance
- Interpret results and draw conclusions

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors:** Ensure all required packages are installed
2. **Data Loading Issues:** Check file paths and data formats
3. **Visualization Problems:** Verify matplotlib backend settings
4. **Memory Issues:** Use smaller datasets for initial testing

### Solutions
```python
# Check package versions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

## 📚 Additional Resources

### Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Tutorials
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

## ✅ Lab Completion Checklist

- [ ] Environment setup completed
- [ ] All exercises executed successfully
- [ ] Data visualizations created
- [ ] Machine learning models implemented
- [ ] Results interpreted and documented
- [ ] Questions answered in notebook

## 📝 Lab Report

Students should document their findings in the Jupyter notebook, including:
- Code implementations
- Data visualizations
- Model results
- Interpretations and insights
- Challenges encountered and solutions

## 🎯 Next Steps

After completing Lab 01, students will be ready for:
- Lab 02: Linear Regression Analysis
- Lab 03: Advanced Linear Regression & Gradient Descent
- Lab 04: Deep Learning & Computer Vision
- Lab 05: Logistic Regression & Classification

---

**Lab Duration:** 3 hours  
**Difficulty Level:** Beginner  
**Prerequisites:** Basic Python, Statistics  
**Deliverables:** Executed Jupyter notebook with results
