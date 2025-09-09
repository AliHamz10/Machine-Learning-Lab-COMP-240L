# Lab 03: Linear Regression on Insurance Dataset - Lab Report

**Subject:** Machine Learning  
**Subject Teacher:** Dr. Abid Ali  
**Lab Supervisor:** Miss. Sana Saleem  
**Student:** [Your Name]  
**Date:** [Current Date]

## Executive Summary

This lab successfully implemented linear regression models on the insurance dataset following all four Lab Practices outlined in the manual. The analysis achieved comprehensive results with multiple model implementations, performance evaluations, and visualizations.

## Dataset Analysis

### Dataset Overview
- **Dataset:** Insurance charges prediction
- **Size:** 1,338 samples with 7 features
- **Features:** age, gender, bmi, children, smoker, region, charges
- **Target:** charges (continuous variable)
- **Missing Values:** 0
- **Outliers Removed:** 22 (1.6% of data)

### Data Preprocessing
1. **Categorical Encoding:** Used LabelEncoder for gender, smoker, and region
2. **Outlier Detection:** Applied Z-score analysis (threshold = 3)
3. **Data Retention:** 98.4% of original data retained after outlier removal

## Lab Practice Results

### Lab Practice 1: Single Variable Linear Regression
- **Feature:** Age only
- **MSE:** 135,891,731.63
- **R²:** 0.1319 (13.19% variance explained)
- **Coefficient:** 239.23 (age impact on charges)
- **Intercept:** 3,916.77

**Analysis:** Age alone explains only 13.19% of the variance in insurance charges, indicating limited predictive power.

### Lab Practice 2: Multi-Variable Linear Regression
- **Features:** All 6 encoded features
- **MSE:** 34,521,053.72 (74.6% reduction from single variable)
- **R²:** 0.7795 (77.95% variance explained)
- **Feature Coefficients:**
  - age: 3,501.62
  - gender_encoded: -208.38
  - bmi: 2,022.28
  - children: 619.21
  - smoker_encoded: 9,607.28 (most important)
  - region_encoded: -435.90

**Analysis:** Multi-variable model significantly outperforms single variable, with smoking status being the most influential factor.

### Lab Practice 3: Binary Classification Metrics
- **Classification:** High vs Low charges (median threshold: $9,412.96)
- **Accuracy:** 0.8788 (87.88%)
- **Precision:** 0.8551 (85.51%)
- **Recall:** 0.9077 (90.77%)
- **F1-Score:** 0.8806 (88.06%)

**Analysis:** Excellent classification performance with balanced precision and recall.

### Lab Practice 4: Custom Gradient Descent
- **MSE:** 34,520,715.99
- **R²:** 0.7795 (matches sklearn implementation)
- **Final Cost:** 18,671,175.94
- **Epochs:** 1,000
- **Learning Rate:** 0.01

**Analysis:** Custom implementation successfully matches sklearn's results, validating the gradient descent algorithm.

## Key Insights

1. **Feature Importance:** Smoking status is the most significant predictor (coefficient: 9,607.28)
2. **Model Performance:** Multi-variable model is 5.9x better than single variable (R²: 0.7795 vs 0.1319)
3. **Data Quality:** Minimal data loss (1.6%) during outlier removal
4. **Algorithm Validation:** Custom gradient descent matches sklearn implementation
5. **Classification:** Strong binary classification performance (87.88% accuracy)

## Generated Visualizations

The following visualizations were created and saved as PNG files:

1. **correlation_matrix.png** - Feature correlation heatmap
2. **feature_distributions.png** - Before/after outlier removal distributions
3. **single_variable_model.png** - Age vs charges regression analysis
4. **multi_variable_model.png** - Multi-feature regression analysis
5. **feature_importance.png** - Feature coefficient importance
6. **gradient_descent_cost.png** - Cost function convergence

## Technical Implementation

### Libraries Used
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
- Jupyter notebook for interactive analysis
- Custom gradient descent implementation

### Code Structure
- **run_analysis.py:** Complete analysis script with all Lab Practices
- **Insurance_Dataset_Linear_Regression.ipynb:** Interactive Jupyter notebook
- **Visualization functions:** Automated plot generation and saving

## Conclusion

This lab successfully demonstrated:
- Comprehensive data preprocessing and analysis
- Multiple linear regression implementations
- Performance metric evaluation
- Custom algorithm development
- Professional visualization generation

The insurance dataset analysis provides valuable insights into factors affecting insurance charges, with smoking status being the most critical predictor. The multi-variable model achieves excellent performance (R² = 0.7795) and can be used for practical insurance charge predictions.

## Files Generated

1. **Insurance_Dataset_Linear_Regression.ipynb** - Complete Jupyter notebook
2. **run_analysis.py** - Analysis script
3. **Lab_03_Report.md** - This comprehensive report
4. **6 PNG visualization files** - All required plots and charts

All objectives from the lab manual have been successfully completed with detailed analysis and professional documentation.
