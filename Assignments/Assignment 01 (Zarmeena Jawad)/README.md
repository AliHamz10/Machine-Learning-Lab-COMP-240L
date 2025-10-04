# Assignment 01: Customer Churn Prediction Analysis

**Class:** BSAI F23 Red  
**Student Number:** B23F0063AI107  
**Due Date:** 04-10-2025

## Business Problem

**Customer Churn Prediction for Telecom Industry**

The telecommunications industry faces significant challenges in customer retention, with high customer acquisition costs and increasing competition leading to substantial revenue losses from customer churn. This analysis aims to develop a machine learning model that can predict customer churn based on demographic and service usage patterns, enabling telecom companies to:

- Proactively identify at-risk customers
- Implement targeted retention strategies
- Optimize customer relationship management
- Reduce customer acquisition costs

## Dataset

**IBM Telco Customer Churn Dataset**

- **Source:** IBM Developer Repository
- **Size:** 7,043 instances
- **Features:** 19 demographic and service features + churn status
- **Target:** Customer churn (Yes/No)

### Features:

1. customerID (removed for analysis)
2. gender
3. SeniorCitizen
4. Partner
5. Dependents
6. tenure
7. PhoneService
8. MultipleLines
9. InternetService
10. OnlineSecurity
11. OnlineBackup
12. DeviceProtection
13. TechSupport
14. StreamingTV
15. StreamingMovies
16. Contract
17. PaperlessBilling
18. PaymentMethod
19. MonthlyCharges
20. TotalCharges

## Methodology

### 1. Data Exploration

- Comprehensive statistical analysis
- Correlation analysis between features
- Churn distribution analysis
- Visual exploratory data analysis

### 2. Data Preprocessing

- Missing value imputation (TotalCharges)
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

The analysis demonstrates that machine learning models can effectively predict customer churn with high accuracy, enabling proactive customer retention and business decision-making.

### Key Findings:

- Best performing model achieves 80.2% accuracy in churn prediction
- Contract type, tenure, and monthly charges are key churn indicators
- Model enables proactive customer retention strategies

## Business Impact

### Applications:

1. **Churn Prevention:** Proactive identification of at-risk customers
2. **Retention Campaigns:** Targeted offers based on churn probability
3. **Customer Segmentation:** Data-driven customer lifecycle management
4. **Resource Allocation:** Focus retention efforts on high-value customers

### Benefits:

- Reduced customer acquisition costs
- Improved customer retention rates
- Enhanced customer relationship management
- Data-driven business decision making

## Files Structure

```
Assignment 01 (Zarmeena Jawad)/
├── assignment01_analysis.py      # Main analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
├── Assignment01_Report.md        # Comprehensive report
├── customer_churn_analysis.png   # Exploratory analysis plots
└── feature_importance_*.png      # Feature importance plots
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
