# Assignment 01: Wine Quality Prediction Analysis

**Class:** BSAI F23 Red  
**Student:** Ali Hamza  
**Student Number:** B23F0063AI107  
**Due Date:** 04-10-2025  
**Course:** Machine Learning Lab (COMP-240L)

## Business Problem

**Wine Quality Prediction for Market Segmentation**

The wine industry faces challenges in consistently predicting wine quality during production, leading to inconsistent pricing strategies and market positioning. This analysis aims to develop a machine learning model that can predict wine quality based on chemical properties, enabling wineries to:

- Automate quality control processes
- Implement data-driven pricing strategies
- Segment products for different market tiers
- Optimize production processes

## Dataset

**Wine Quality Dataset (Red Wine)**

- **Source:** UCI Machine Learning Repository
- **Size:** 1,599 instances (meets 1000+ requirement)
- **Features:** 11 chemical properties + quality score (meets 10+ requirement)
- **Target:** Wine quality (3-8 scale, categorized as Low/Medium/High)
- **Data Quality:** Clean dataset with no missing values

### Features:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

## Methodology

### 1. Data Exploration

- Comprehensive statistical analysis
- Correlation analysis between features
- Quality distribution analysis
- Visual exploratory data analysis

### 2. Data Preprocessing

- Quality categorization (Low: 3-5, Medium: 6, High: 7-8)
- Feature scaling for algorithms requiring normalization
- Train-test split (80-20) with stratification

### 3. Model Selection

Four machine learning algorithms implemented:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

### 4. Model Evaluation

- Accuracy, Precision, Recall, F1-Score
- 5-fold cross-validation
- Confusion matrix analysis
- Feature importance analysis

## Results

The analysis demonstrates that machine learning models can effectively predict wine quality with high accuracy, enabling automated quality assessment and business decision-making.

### Model Performance:

- **Best Model:** Random Forest Classifier
- **Accuracy:** 80.2%
- **Precision:** 75.3% (High Quality)
- **Recall:** 68.9% (High Quality)
- **F1-Score:** 72.0%

### Key Findings:

- Random Forest achieves highest accuracy in quality prediction
- Alcohol content and volatile acidity are most important features
- Chemical properties show strong correlation with wine quality
- Model enables automated quality control and pricing strategies

## Business Impact

### Applications:

1. **Quality Control:** Automated wine quality assessment during production
2. **Pricing Strategy:** Data-driven pricing based on predicted quality
3. **Market Segmentation:** Target different customer segments effectively
4. **Inventory Management:** Optimize stock based on quality predictions

### Benefits:

- Reduced manual tasting costs
- Consistent quality standards
- Improved pricing accuracy
- Enhanced market positioning

## Files Structure

```
Assignment 01 (Ali Hamza)/
├── assignment01_analysis.py                    # Main analysis script (591 lines)
├── Assignment01_Report.md                      # Comprehensive report (394 lines)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This documentation
├── wine_quality_analysis.png                   # 9-panel exploratory analysis
├── confusion_matrix.png                        # Model confusion matrix
├── feature_importance_random_forest.png        # Feature importance plot
├── model_comparison.png                        # Model performance comparison
├── model_results.csv                           # Detailed performance metrics
├── detailed_results.txt                        # Analysis summary
├── processed_adult_income.csv                  # Processed dataset
├── COMP 240 Machine Learning Assignment 01.pdf # PDF report
└── assignment01_env/                           # Virtual environment
    ├── bin/                                    # Python executables
    ├── lib/                                    # Installed packages
    └── pyvenv.cfg                              # Environment config
```

## Usage

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis:**

   ```bash
   python assignment01_analysis.py
   ```

3. **View Results:**
   - Generated plots will be saved as PNG files
   - Console output shows detailed analysis results

## Technical Requirements Met

- ✅ Dataset with 10+ features and 1000+ instances
- ✅ Real-world business problem identification
- ✅ Comprehensive data exploration
- ✅ Data preprocessing and feature engineering
- ✅ Multiple ML algorithm implementation
- ✅ Model evaluation and comparison
- ✅ Business insights and recommendations
- ✅ Clean, documented Python code

## Academic Integrity

This work is original and follows academic integrity guidelines. All analysis, code, and insights are independently developed for this assignment.

## References

[IEEE style references will be added in the final report]
