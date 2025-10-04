# Assignment 01: Income Prediction Analysis

**Class:** BSAI F23 Red  
**Student:** Zarmeena Jawad  
**Student Number:** B23F0063AI107  
**Due Date:** 04-10-2025  
**Course:** Machine Learning Lab (COMP-240L)

## Business Problem

**Income Prediction for Financial Services Industry**

Financial institutions face significant challenges in accurate income assessment for credit decisions, loan approvals, and financial planning services. The lack of reliable income prediction methods results in inconsistent credit assessment processes, high risk of loan defaults, and difficulty in identifying high-income customers. This analysis aims to develop a machine learning model that can predict individual income levels based on demographic and economic factors, enabling financial institutions to:

- Improve credit assessment accuracy
- Optimize loan approval processes
- Enhance financial planning services
- Reduce credit risk and defaults

## Dataset

**UCI Adult Income Dataset**

- **Source:** UCI Machine Learning Repository
- **Size:** 32,561 instances (exceeds 10,000 requirement)
- **Features:** 14 demographic and economic features + income level
- **Target:** Income level (<=50K or >50K)
- **Data Quality:** Clean dataset with no missing values

### Features:

1. age - Age in years
2. workclass - Type of employment
3. fnlwgt - Final weight
4. education - Education level
5. education-num - Education (numerical)
6. marital-status - Marital status
7. occupation - Job occupation
8. relationship - Family relationship
9. race - Race/ethnicity
10. sex - Gender
11. capital-gain - Capital gains
12. capital-loss - Capital losses
13. hours-per-week - Hours worked per week
14. native-country - Country of origin

## Methodology

### 1. Data Exploration

- Comprehensive statistical analysis
- Correlation analysis between features
- Income distribution analysis
- Visual exploratory data analysis

### 2. Data Preprocessing

- Missing value handling (removed 24 duplicates)
- Categorical variable encoding using LabelEncoder
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

The analysis demonstrates that machine learning models can effectively predict income levels with high accuracy, enabling data-driven financial decision making and business applications.

### Model Performance:
- **Best Model:** Gradient Boosting Classifier
- **Accuracy:** 87.0%
- **Precision:** 79.7% (High Income)
- **Recall:** 61.5% (High Income)
- **F1-Score:** 69.4%

### Key Findings:
- Gradient Boosting achieves highest accuracy in income prediction
- Relationship status, capital gains, and education are most important features
- Model enables accurate credit assessment and financial planning

## Business Impact

### Applications:

1. **Credit Assessment:** Accurate income prediction for loan decisions
2. **Financial Planning:** Personalized advice based on predicted income
3. **Risk Management:** Better assessment of credit risk and repayment capacity
4. **Customer Segmentation:** Data-driven targeting for financial products

### Benefits:

- Improved credit risk assessment
- Reduced loan default rates
- Enhanced financial planning services
- Data-driven business decision making

## Files Structure

```
Assignment 01 (Zarmeena Jawad)/
├── assignment01_analysis.py                           # Main analysis script (595 lines)
├── Assignment01_Report.md                             # Comprehensive report (394 lines)
├── requirements.txt                                   # Python dependencies
├── README.md                                          # This documentation
├── adult_income_analysis.png                          # 9-panel exploratory analysis
├── adult_income_detailed_analysis.png                 # Statistical analysis plots
├── feature_importance_gradient_boosting.png           # Feature importance plot
├── model_evaluation_gradient_boosting.png             # Model evaluation plots
├── model_results.csv                                  # Detailed performance metrics
├── detailed_results.txt                               # Analysis summary
└── assignment01_env/                                  # Virtual environment
    ├── bin/                                           # Python executables
    ├── lib/                                           # Installed packages
    └── pyvenv.cfg                                     # Environment config
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

[IEEE style references included in the comprehensive report]
