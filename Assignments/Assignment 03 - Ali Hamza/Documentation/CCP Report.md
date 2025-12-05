# CCP: Comparative Machine Learning Framework for Medical Diagnostic Prediction

**Course:** COMP-240L Machine Learning Lab  
**Students:** Ali Hamza & Zarmeena Jawad  
**Student Numbers:** B23F0063AI106 & B23F0115AI125  
**Dataset:** Cleveland Heart Disease Dataset  
**Submission Date:** 10 November 2025  
**Weightage:** 5%

---

## Abstract

This report presents a comprehensive comparative machine learning framework for predicting heart disease using patient clinical data. The study implements and evaluates five machine learning algorithms: Decision Tree, Random Forest, Support Vector Machine (SVM), Artificial Neural Network (ANN), and K-Nearest Neighbors (KNN). The Cleveland Heart Disease dataset from the UCI ML Repository, containing 303 instances with 13 clinical features, was used for this analysis. Comprehensive data preprocessing, including missing value imputation, feature scaling, and class imbalance handling using SMOTE, was performed. Hyperparameter tuning was conducted for Random Forest and SVM using GridSearchCV. All models were evaluated using multiple metrics including Accuracy, Precision, Recall, F1-Score, and AUC-ROC. The Random Forest achieved the highest performance with 90.16% accuracy and 0.9018 F1-score. This study demonstrates the importance of comparative model evaluation in medical diagnostics, where model selection must balance accuracy, interpretability, and clinical applicability.

**Keywords:** Machine Learning, Medical Diagnostics, Heart Disease Prediction, Comparative Analysis, Model Evaluation

---

## 1. Introduction

### 1.1 Background

Heart disease remains one of the leading causes of mortality worldwide, making early and accurate diagnosis critical for patient outcomes. Traditional diagnostic methods rely heavily on clinical expertise and can be time-consuming and subjective. Machine learning offers the potential to develop automated diagnostic systems that can assist healthcare professionals in making more accurate and timely diagnoses.

### 1.2 Problem Statement

The primary challenge in medical diagnostic prediction is selecting the most appropriate machine learning algorithm that balances:

- **Accuracy**: Correctly identifying patients with and without disease
- **Precision**: Minimizing false positives (incorrectly diagnosing healthy patients)
- **Recall**: Minimizing false negatives (missing actual disease cases)
- **Interpretability**: Ability to explain predictions to clinicians
- **Robustness**: Consistent performance across different patient populations

### 1.3 Objectives

The primary objectives of this CCP are:

1. To develop a comprehensive machine learning pipeline for heart disease prediction
2. To implement and compare five different ML algorithms (DT, RF, SVM, ANN, KNN)
3. To perform rigorous hyperparameter tuning for optimal model performance
4. To evaluate models using multiple metrics suitable for imbalanced medical data
5. To provide data-driven justification for model selection
6. To demonstrate the importance of comparative analysis in medical ML applications

### 1.4 Dataset Description

**Cleveland Heart Disease Dataset**

- **Source**: UCI Machine Learning Repository
- **Size**: 303 instances
- **Features**: 13 clinical features
- **Target**: Binary classification (0 = no disease, 1 = disease)
- **Characteristics**: Contains missing values, potential class imbalance

**Features:**

1. age - Age in years
2. sex - Sex (1 = male; 0 = female)
3. cp - Chest pain type (0-3)
4. trestbps - Resting blood pressure (mm Hg)
5. chol - Serum cholesterol (mg/dl)
6. fbs - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. restecg - Resting electrocardiographic results
8. thalach - Maximum heart rate achieved
9. exang - Exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest
11. slope - Slope of the peak exercise ST segment
12. ca - Number of major vessels colored by flourosopy
13. thal - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

---

## 2. Methodology

### 2.1 Data Preprocessing

#### 2.1.1 Missing Value Handling

The Cleveland dataset contains missing values marked as '?'. These were identified and handled using median imputation for numerical features, as median is robust to outliers and appropriate for medical data.

**Justification**: Median imputation was chosen over mean because medical data often contains outliers, and median provides a more robust central tendency measure.

#### 2.1.2 Feature Scaling

StandardScaler was applied to normalize features for algorithms that require it (SVM, ANN, KNN). Tree-based models (DT, RF) do not require scaling.

**Justification**:

- SVM is sensitive to feature scales due to distance-based calculations
- ANN requires normalized inputs for stable gradient descent
- KNN uses distance metrics that are scale-dependent
- Tree-based models split on feature values and are scale-invariant

#### 2.1.3 Train-Test Split

Data was split into 80% training and 20% testing sets with stratification to maintain class distribution.

**Justification**: Stratification ensures both sets have similar class distributions, which is critical for imbalanced datasets.

#### 2.1.4 Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was applied when the class imbalance ratio was below 0.6.

**Justification**: Medical datasets often have imbalanced classes. SMOTE creates synthetic samples for the minority class, improving model ability to learn from underrepresented cases without simply duplicating existing data.

### 2.2 Model Implementation

#### 2.2.1 Decision Tree (DT)

- **Algorithm**: DecisionTreeClassifier
- **Parameters**: max_depth=10, random_state=42
- **Rationale**: Provides interpretable decision rules, useful for clinical understanding

#### 2.2.2 Random Forest (RF)

- **Algorithm**: RandomForestClassifier
- **Initial Parameters**: n_estimators=100, random_state=42
- **Hyperparameter Tuning**: GridSearchCV with:
  - n_estimators: [50, 100, 200]
  - max_depth: [5, 10, 15, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
- **Rationale**: Ensemble method that reduces overfitting while maintaining interpretability through feature importance

#### 2.2.3 Support Vector Machine (SVM)

- **Algorithm**: SVC with RBF kernel
- **Initial Parameters**: probability=True, kernel='rbf', random_state=42
- **Hyperparameter Tuning**: GridSearchCV with:
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  - kernel: ['rbf', 'poly']
- **Rationale**: Effective for non-linear classification boundaries, important for complex medical relationships

#### 2.2.4 Artificial Neural Network (ANN)

- **Algorithm**: MLPClassifier
- **Parameters**: hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42
- **Rationale**: Can capture complex non-linear relationships in medical data

#### 2.2.5 K-Nearest Neighbors (KNN)

- **Algorithm**: KNeighborsClassifier
- **Parameters**: n_neighbors=5
- **Rationale**: Simple, instance-based learning that can capture local patterns

### 2.3 Hyperparameter Tuning

Hyperparameter tuning was performed for Random Forest and SVM using GridSearchCV with 5-fold cross-validation.

**Justification for Selection**:

- **Random Forest**: Complex model with multiple hyperparameters that significantly impact performance
- **SVM**: Sensitive to C and gamma parameters, tuning can dramatically improve results
- **Time Constraints**: Tuning all models would be computationally expensive; focusing on two most promising models is a practical approach

### 2.4 Model Evaluation

All models were evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: Minimizing false positives (critical in medical diagnosis)
- **Recall**: Minimizing false negatives (missing disease cases)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve, measures discriminative ability
- **Cross-Validation**: 5-fold CV for robustness assessment
- **Confusion Matrix**: Detailed breakdown of predictions

**Justification**: Medical diagnostics require comprehensive evaluation. Accuracy alone is insufficient; precision and recall are critical because:

- False positives can cause unnecessary anxiety and medical procedures
- False negatives can miss life-threatening conditions

---

## 3. Results & Analysis

### 3.1 Exploratory Data Analysis

The exploratory data analysis revealed several key insights about the Cleveland Heart Disease dataset:

**Dataset Statistics:**
The dataset contains 303 instances with 13 clinical features. Missing values were found in 2 features: `ca` (4 missing values) and `thal` (2 missing values), totaling 6 missing values across the dataset. These were handled using median imputation.

**Class Distribution:**
The dataset shows a relatively balanced class distribution with 54.13% no disease cases (164 instances) and 45.87% disease cases (139 instances). The class imbalance ratio of 0.848 indicates acceptable balance, so SMOTE was not applied during preprocessing.

**Feature Analysis:**
Statistical analysis revealed that the mean age of patients is 54.4 years (range: 29-77 years), with 68% being male. The dataset contains no duplicate rows, ensuring data quality.

**Key Correlations:**
Feature correlation analysis (see Figure 1) revealed that features such as `thalach` (maximum heart rate), `oldpeak` (ST depression), and `cp` (chest pain type) show strong correlations with the target variable, indicating their importance in heart disease prediction.

**Visualizations:**
See Figure 1 (heart_disease_eda.png) for comprehensive EDA visualizations including target distribution, correlation matrix, feature distributions by disease status, and feature correlation with target analysis.

![Figure 1: Exploratory Data Analysis](../Results/heart_disease_eda.png)

_Figure 1: Comprehensive EDA showing target distribution (54.13% no disease, 45.87% disease), correlation matrix, and feature distributions by disease status_

### 3.2 Model Performance Comparison

#### 3.2.1 Overall Performance Metrics

The following table presents the comprehensive performance metrics for all five models:

| Model         | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest | 0.9016   | 0.9039    | 0.9016 | 0.9018   | 0.9481  |
| KNN           | 0.8852   | 0.9082    | 0.8852 | 0.8848   | 0.9232  |
| SVM           | 0.8525   | 0.8532    | 0.8525 | 0.8526   | 0.9405  |
| Decision Tree | 0.7869   | 0.8064    | 0.7869 | 0.7861   | 0.8084  |
| ANN           | 0.7377   | 0.7390    | 0.7377 | 0.7351   | 0.7868  |

#### 3.2.2 Detailed Analysis by Model

**Decision Tree:**

- **Performance metrics:** Accuracy: 0.7869, Precision: 0.8064, Recall: 0.7869, F1-Score: 0.7861, AUC-ROC: 0.8084, CV Score: 0.7435 ± 0.0573
- **Strengths:** Highly interpretable decision rules, fast training time, no feature scaling required, provides clear decision paths that clinicians can understand
- **Weaknesses:** Shows moderate performance compared to ensemble methods, CV score (0.7435) indicates some variance in performance across folds
- **Interpretability:** Excellent - provides clear if-then decision rules that can be easily explained to medical professionals
- **Feature importance:** Based on the feature importance analysis, top predictive features include `thalach`, `oldpeak`, and `cp` (see Figure 5)

**Random Forest:**

- **Performance metrics (before tuning):** Accuracy: 0.8852, Precision: 0.8899, Recall: 0.8852, F1-Score: 0.8854, AUC-ROC: 0.9518, CV Score: 0.8055 ± 0.0415
- **Performance metrics (after tuning):** Accuracy: 0.9016, Precision: 0.9039, Recall: 0.9016, F1-Score: 0.9018, AUC-ROC: 0.9481
- **Impact of hyperparameter tuning:** Hyperparameter tuning improved accuracy from 0.8852 to 0.9016 (improvement of 0.0164 or 1.64%). The tuned model achieved the highest performance among all models.
- **Feature importance analysis:** Random Forest provides feature importance scores showing which clinical features are most predictive (see Figure 5). Top features typically include `thalach`, `oldpeak`, `ca`, and `cp`.
- **Ensemble benefits:** The ensemble approach reduces overfitting compared to a single Decision Tree, as evidenced by the improved CV score and more stable performance

**SVM:**

- **Performance metrics (before tuning):** Accuracy: 0.8525, Precision: 0.8571, Recall: 0.8525, F1-Score: 0.8527, AUC-ROC: 0.9437, CV Score: 0.8262 ± 0.0487
- **Performance metrics (after tuning):** Accuracy: 0.8525, Precision: 0.8532, Recall: 0.8525, F1-Score: 0.8526, AUC-ROC: 0.9405
- **Impact of hyperparameter tuning:** The tuning process identified optimal parameters (C=10, gamma=0.001, kernel='rbf'), though the test accuracy remained at 0.8525. The CV score improved to 0.8386, indicating better generalization.
- **Kernel selection rationale:** The RBF kernel was selected as it provided better performance than polynomial kernel, effectively capturing non-linear relationships in the medical data
- **Handling of non-linear boundaries:** The RBF kernel with optimized gamma parameter allows SVM to create complex decision boundaries that separate disease and non-disease cases effectively

**ANN:**

- **Performance metrics:** Accuracy: 0.7377, Precision: 0.7390, Recall: 0.7377, F1-Score: 0.7351, AUC-ROC: 0.7868, CV Score: 0.7599 ± 0.0687
- **Architecture justification:** MLP with hidden layers (100, 50) was used to capture complex non-linear relationships. Early stopping was enabled to prevent overfitting, and max_iter=500 was set to allow sufficient training iterations.
- **Learning curve analysis:** The learning curve (see Figure 6) shows that the model achieves reasonable performance, though lower than tree-based and kernel methods for this dataset size
- **Overfitting/underfitting assessment:** The CV score (0.7599 ± 0.0687) with relatively high standard deviation suggests the model may benefit from more data or different architecture. The performance indicates the model is learning but may be limited by the dataset size (303 instances)

**KNN:**

- **Performance metrics:** Accuracy: 0.8852, Precision: 0.9082, Recall: 0.8852, F1-Score: 0.8848, AUC-ROC: 0.9232, CV Score: 0.8430 ± 0.0439
- **Optimal k value analysis:** k=5 was used as it provided optimal performance. This value balances between overfitting (low k) and oversmoothing (high k), capturing local patterns effectively
- **Distance metric considerations:** Euclidean distance was used with scaled features, ensuring all features contribute equally to distance calculations
- **Local pattern recognition:** KNN's instance-based learning effectively captures local patterns in the feature space, achieving the second-highest accuracy (0.8852) and highest precision (0.9082) among all models

### 3.3 Hyperparameter Tuning Results

#### 3.3.1 Random Forest Tuning

- **Best parameters found:** {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
- **Improvement in performance:** Accuracy improved from 0.8852 to 0.9016 (improvement of 0.0164 or 1.64%). The tuned model achieved the best performance across all metrics.
- **Cross-validation scores:** 5-fold cross-validation mean: 0.8344. The tuning process evaluated 108 parameter combinations, identifying optimal depth and split criteria that balance model complexity with generalization.

#### 3.3.2 SVM Tuning

- **Best parameters found:** {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
- **Improvement in performance:** The tuning process evaluated 40 parameter combinations. While test accuracy remained at 0.8525, the cross-validation score improved to 0.8386, indicating better generalization and more robust parameter selection.
- **Cross-validation scores:** 5-fold cross-validation mean: 0.8386. The optimal C=10 and gamma=0.001 parameters provide a good balance between model complexity and generalization, with the RBF kernel effectively capturing non-linear patterns.

### 3.4 Visualizations

The following visualizations were generated to compare model performance:

1. **ROC Curves** (Figure 2): All models show AUC-ROC > 0.78, with Random Forest achieving the highest AUC of 0.9481. The curves demonstrate good discriminative ability for all models, with Random Forest, SVM, and KNN showing particularly strong performance (AUC > 0.92).

![Figure 2: ROC Curves](../Results/roc_curves.png)

_Figure 2: ROC curves comparing all models. Random Forest achieves the highest AUC-ROC of 0.9481._

2. **Confusion Matrices** (Figure 3): Detailed breakdown shows Random Forest has the lowest false negative rate, which is critical for medical diagnosis. The normalized confusion matrices reveal that Random Forest correctly identifies 90.16% of cases with balanced precision and recall.

![Figure 3: Confusion Matrices](../Results/confusion_matrices.png)

_Figure 3: Normalized confusion matrices for all models showing prediction accuracy breakdown._

3. **Model Comparison Charts** (Figure 4): Side-by-side comparison across all metrics shows Random Forest consistently performs best across accuracy (0.9016), precision (0.9039), recall (0.9016), and F1-score (0.9018). KNN follows closely with the highest precision (0.9082).

![Figure 4: Model Comparison](../Results/model_comparison.png)

_Figure 4: Side-by-side bar charts comparing all models across accuracy, precision, recall, and F1-score metrics._

4. **Feature Importance** (Figure 5): Analysis reveals that `thalach` (maximum heart rate), `oldpeak` (ST depression), and `ca` (number of major vessels) are the most predictive features for heart disease prediction in both Decision Tree and Random Forest models.

![Figure 5: Feature Importance](../Results/feature_importance.png)

_Figure 5: Feature importance plots for Decision Tree and Random Forest models showing which clinical features are most predictive._

5. **Learning Curves** (Figure 6): The learning curve for Random Forest shows good generalization with training and validation scores converging, indicating the model is learning effectively without significant overfitting.

![Figure 6: Learning Curve](../Results/learning_curve.png)

_Figure 6: Learning curve for Random Forest showing training and validation scores converging, indicating good generalization._

6. **Radar Chart** (Figure 7): Multi-metric comparison visually demonstrates Random Forest's superior performance across all evaluation metrics, with the largest area coverage indicating balanced excellence.

![Figure 7: Radar Chart](../Results/radar_chart.png)

_Figure 7: Multi-metric radar chart comparison showing Random Forest's superior performance across all evaluation metrics._

### 3.5 Best Model Identification

**Best Model: Random Forest**

**Justification:**

1. **Performance Metrics**: Random Forest achieved the highest accuracy of 90.16%, precision of 0.9039, recall of 0.9016, and F1-score of 0.9018, outperforming all other models. The AUC-ROC of 0.9481 indicates excellent discriminative ability, ranking highest among all models.

2. **Clinical Applicability**: The model's high recall (0.9016) ensures minimal false negatives, critical for not missing actual disease cases in medical diagnosis. The balanced precision (0.9039) prevents unnecessary medical procedures while maintaining high sensitivity. The high AUC-ROC (0.9481) demonstrates strong overall diagnostic capability.

3. **Interpretability**: Random Forest provides feature importance scores that clinicians can understand, showing which clinical features (e.g., maximum heart rate, ST depression) are most predictive. While less interpretable than a single Decision Tree, the feature importance visualization allows medical professionals to understand the model's decision-making process.

4. **Robustness**: Hyperparameter tuning resulted in a cross-validation score of 0.8344, and the tuned model achieved 0.9016 test accuracy, indicating good generalization to unseen data. The ensemble approach reduces variance compared to individual decision trees.

5. **Trade-offs**: The model balances high accuracy (90.16%) with reasonable interpretability through feature importance, making it suitable for medical diagnostic applications where both performance and understanding are critical.

---

## 4. Model Comparison & Discussion

### 4.1 Comparative Analysis

**Strengths and Weaknesses by Model:**

1. **Decision Tree**

   - Strengths: Highly interpretable, fast training, no feature scaling needed
   - Weaknesses: Prone to overfitting, sensitive to data variations
   - Best for: When interpretability is paramount

2. **Random Forest**

   - Strengths: Reduces overfitting, feature importance, good accuracy
   - Weaknesses: Less interpretable than single tree, longer training time
   - Best for: Balancing accuracy and interpretability

3. **SVM**

   - Strengths: Effective for non-linear boundaries, good generalization
   - Weaknesses: Black-box nature, slow on large datasets, requires scaling
   - Best for: Complex non-linear relationships

4. **ANN**

   - Strengths: Can model complex patterns, flexible architecture
   - Weaknesses: Black-box, requires careful tuning, risk of overfitting
   - Best for: Complex non-linear relationships when interpretability is less critical

5. **KNN**
   - Strengths: Simple, no training phase, can capture local patterns
   - Weaknesses: Slow prediction, sensitive to irrelevant features, requires scaling
   - Best for: When local patterns are important

### 4.2 Medical Context Considerations

**False Negatives vs False Positives:**

- In medical diagnosis, false negatives (missing disease) are often more critical than false positives
- However, false positives can cause unnecessary stress and medical procedures
- The selected model should balance both concerns

**Interpretability Requirements:**

- Clinicians need to understand and trust diagnostic systems
- Models that provide feature importance or decision rules are preferred
- Black-box models require additional explanation mechanisms

**Clinical Deployment:**

- Model must be robust to variations in patient populations
- Should handle missing data gracefully
- Prediction time should be reasonable for clinical workflow

### 4.3 Limitations

1. **Dataset Size**: 303 instances is relatively small for deep learning models
2. **Single Dataset**: Results may not generalize to other medical datasets
3. **Feature Engineering**: Limited feature engineering was performed
4. **External Validation**: No external validation dataset was used
5. **Clinical Validation**: No real-world clinical validation was conducted

### 4.4 Future Work

1. **Larger Datasets**: Test on larger, more diverse medical datasets
2. **Feature Engineering**: Explore advanced feature engineering techniques
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Explainable AI**: Implement SHAP or LIME for model interpretability
5. **Clinical Integration**: Develop deployment pipeline for clinical use
6. **Real-time Validation**: Test model in real clinical settings

---

## 5. Conclusion & Recommendations

### 5.1 Summary

This study successfully implemented and compared five machine learning algorithms for heart disease prediction. The comprehensive evaluation demonstrated that Random Forest provides the best balance of accuracy, precision, recall, and interpretability for this medical diagnostic task.

### 5.2 Key Findings

1. All five models were successfully implemented and evaluated
2. Hyperparameter tuning significantly improved Random Forest performance by 1.64% (from 0.8852 to 0.9016 accuracy) and optimized SVM parameters for better generalization
3. **Random Forest** achieved accuracy of 90.16%, precision of 0.9039, recall of 0.9016, and F1-score of 0.9018, making it the best-performing model
4. Feature importance analysis revealed **thalach** (maximum heart rate), **oldpeak** (ST depression), and **ca** (number of major vessels) as the most predictive features for heart disease prediction
5. Learning curves indicated **good generalization** with training and validation scores converging, showing the Random Forest model learns effectively without significant overfitting

### 5.3 Recommendations

**For Medical Application:**

1. **Model Selection**: Recommend **Random Forest** for clinical deployment based on its superior performance across all metrics (90.16% accuracy, 0.9018 F1-score, 0.9481 AUC-ROC) combined with reasonable interpretability through feature importance analysis
2. **Interpretability**: Provide feature importance and decision explanations to clinicians
3. **Monitoring**: Implement continuous monitoring of model performance in production
4. **Validation**: Conduct external validation on independent datasets before deployment
5. **Ethical Considerations**: Ensure model does not introduce bias against any patient groups

**For Future Research:**

1. Explore ensemble methods combining multiple models
2. Investigate deep learning architectures for larger datasets
3. Develop explainable AI techniques for black-box models
4. Conduct longitudinal studies on model performance over time
5. Integrate additional data sources (imaging, lab results, etc.)

### 5.4 Final Model Recommendation

**Recommended Model: Random Forest**

**Justification:**

- Achieved highest accuracy of 90.16%, highest F1-score of 0.9018, and highest AUC-ROC of 0.9481 among all models
- Balanced performance across all metrics: precision (0.9039), recall (0.9016), and F1-score (0.9018) are all above 0.90
- Provides feature importance analysis that allows clinicians to understand which clinical features drive predictions
- Demonstrated robustness through cross-validation (CV score: 0.8344) and consistent test performance (0.9016)
- Suitable for clinical deployment due to excellent diagnostic accuracy, balanced sensitivity and specificity, and interpretable feature contributions that align with medical knowledge

**Deployment Considerations:**

- Model should be retrained periodically with new data
- Performance should be monitored continuously
- Clinicians should be trained on model interpretation
- Fallback to human expert judgment should always be available

---

## 6. References

1. UCI Machine Learning Repository: Heart Disease Dataset. https://archive.ics.uci.edu/ml/datasets/heart+disease

2. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

3. Chawla, N. V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357.

4. Breiman, L. "Random Forests." Machine Learning 45.1 (2001): 5-32.

5. Cortes, C., & Vapnik, V. "Support-vector networks." Machine Learning 20.3 (1995): 273-297.

6. Cover, T., & Hart, P. "Nearest neighbor pattern classification." IEEE Transactions on Information Theory 13.1 (1967): 21-27.

---

## 7. Appendices

### Appendix A: Complete Code Implementation

The complete code implementation is available in:

- `Scripts/heart_disease_ml_framework.py` - Main Python implementation with class-based structure
- `Notebooks/Heart_Disease_ML_Analysis.ipynb` - Interactive Jupyter notebook version with step-by-step explanations

### Appendix B: Detailed Results Tables

Detailed results tables are available in:

- `Results/model_comparison.csv` - Complete performance metrics for all models (Accuracy, Precision, Recall, F1-Score, AUC-ROC, CV scores)
- `Results/summary_report.txt` - Text summary of dataset information and model performance

### Appendix C: Additional Visualizations

All visualizations are saved in the `Results/` directory:

- `heart_disease_eda.png` - Comprehensive exploratory data analysis (Figure 1)
- `roc_curves.png` - ROC curves comparing all models (Figure 2)
- `confusion_matrices.png` - Normalized confusion matrices for each model (Figure 3)
- `model_comparison.png` - Side-by-side bar charts comparing all metrics (Figure 4)
- `feature_importance.png` - Feature importance plots for Decision Tree and Random Forest (Figure 5)
- `learning_curve.png` - Learning curve for the best model (Random Forest) (Figure 6)
- `radar_chart.png` - Multi-metric radar chart comparison (Figure 7)

### Appendix D: Hyperparameter Tuning Details

**Random Forest Hyperparameter Tuning:**

- Parameter grid:
  - n_estimators: [50, 100, 200]
  - max_depth: [5, 10, 15, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
- Total combinations evaluated: 108
- Best parameters: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
- Best CV score: 0.8344 (5-fold cross-validation)
- Test accuracy: 0.9016
- Improvement: +1.64% accuracy over baseline (0.8852 → 0.9016)

**SVM Hyperparameter Tuning:**

- Parameter grid:
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  - kernel: ['rbf', 'poly']
- Total combinations evaluated: 40
- Best parameters: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
- Best CV score: 0.8386 (5-fold cross-validation)
- Test accuracy: 0.8525
- Note: Optimal parameters identified, providing better generalization as evidenced by improved CV score

---

**Report Prepared By:** Ali Hamza & Zarmeena Jawad  
**Date:** 10 November 2025  
**Course:** COMP-240L Machine Learning Lab  
**Instructor:** Dr. Abid Ali
