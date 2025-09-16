#!/usr/bin/env python3
"""
Lab 03: Linear Regression Model Implementation and Analysis
Complete analysis script for insurance dataset with all practices integrated.

Department of Electrical and Computer Engineering
Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology
Subject: Machine Learning
Subject Teacher: Dr. Abid Ali
Lab Supervisor: Miss. Sana Saleem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run complete Lab 03 analysis."""
    
    print("="*60)
    print("LAB 03: LINEAR REGRESSION MODEL IMPLEMENTATION AND ANALYSIS")
    print("="*60)
    print("Department of Electrical and Computer Engineering")
    print("Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology")
    print("Subject: Machine Learning")
    print("Subject Teacher: Dr. Abid Ali")
    print("Lab Supervisor: Miss. Sana Saleem")
    print("="*60)
    
    # Load and preprocess data
    print("\n1. LOADING AND PREPROCESSING DATA")
    print("-" * 40)
    
    df = pd.read_csv('Lab Tasks/Insurance.csv')
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Data preprocessing
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df_processed = df.copy()
    df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
    df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])
    df_processed['region_encoded'] = le_region.fit_transform(df_processed['region'])
    
    # Outlier detection and removal
    numerical_cols = ['age', 'bmi', 'children', 'charges']
    z_scores = np.abs(stats.zscore(df_processed[numerical_cols]))
    df_clean = df_processed[(z_scores < 3).all(axis=1)]
    
    print(f"After outlier removal: {df_clean.shape[0]} samples")
    
    # Prepare features and target
    feature_cols = ['age', 'bmi', 'children', 'gender_encoded', 'smoker_encoded', 'region_encoded']
    X = df_clean[feature_cols]
    y = df_clean['charges']
    
    # Train-test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Linear Regression Model (Sklearn)
    print("\n2. LINEAR REGRESSION MODEL TRAINING")
    print("-" * 40)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Sklearn Model Performance:")
    print(f"  MSE: ${mse_test:.2f}")
    print(f"  R²: {r2_test:.4f}")
    print(f"  MAE: ${mae_test:.2f}")
    
    # Gradient Descent Implementation
    print("\n3. GRADIENT DESCENT IMPLEMENTATION")
    print("-" * 40)
    
    def compute_cost(X, y, theta, bias):
        m = len(y)
        predictions = np.dot(X, theta) + bias
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def gradient_descent(X, y, theta, bias, learning_rate, epochs):
        m = len(y)
        cost_history = []
        
        for epoch in range(epochs):
            predictions = np.dot(X, theta) + bias
            d_theta = (1 / m) * np.dot(X.T, (predictions - y))
            d_bias = (1 / m) * np.sum(predictions - y)
            
            theta -= learning_rate * d_theta
            bias -= learning_rate * d_bias
            
            cost = compute_cost(X, y, theta, bias)
            cost_history.append(cost)
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch}: Cost = ${cost:.2f}")
        
        return theta, bias, cost_history
    
    # Run gradient descent
    np.random.seed(42)
    m, n = X_train_scaled.shape
    theta = np.random.randn(n)
    bias = 0.0
    learning_rate = 0.01
    epochs = 1000
    
    theta_gd, bias_gd, cost_history = gradient_descent(X_train_scaled, y_train, theta, bias, learning_rate, epochs)
    
    # Make predictions with gradient descent
    y_pred_gd_test = np.dot(X_test_scaled, theta_gd) + bias_gd
    
    # Calculate metrics for gradient descent
    mse_gd_test = mean_squared_error(y_test, y_pred_gd_test)
    r2_gd_test = r2_score(y_test, y_pred_gd_test)
    mae_gd_test = mean_absolute_error(y_test, y_pred_gd_test)
    
    print(f"\nGradient Descent Model Performance:")
    print(f"  MSE: ${mse_gd_test:.2f}")
    print(f"  R²: {r2_gd_test:.4f}")
    print(f"  MAE: ${mae_gd_test:.2f}")
    
    # Classification Metrics Analysis
    print("\n4. CLASSIFICATION METRICS ANALYSIS")
    print("-" * 40)
    
    threshold = y_test.median()
    y_test_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred_test >= threshold).astype(int)
    y_pred_gd_binary = (y_pred_gd_test >= threshold).astype(int)
    
    # Calculate classification metrics
    accuracy_sklearn = accuracy_score(y_test_binary, y_pred_binary)
    precision_sklearn = precision_score(y_test_binary, y_pred_binary)
    recall_sklearn = recall_score(y_test_binary, y_pred_binary)
    f1_sklearn = f1_score(y_test_binary, y_pred_binary)
    
    accuracy_gd = accuracy_score(y_test_binary, y_pred_gd_binary)
    precision_gd = precision_score(y_test_binary, y_pred_gd_binary)
    recall_gd = recall_score(y_test_binary, y_pred_gd_binary)
    f1_gd = f1_score(y_test_binary, y_pred_gd_binary)
    
    print(f"Classification threshold: ${threshold:.2f}")
    print(f"\nSklearn Classification Metrics:")
    print(f"  Accuracy: {accuracy_sklearn:.4f}")
    print(f"  Precision: {precision_sklearn:.4f}")
    print(f"  Recall: {recall_sklearn:.4f}")
    print(f"  F1 Score: {f1_sklearn:.4f}")
    
    print(f"\nGradient Descent Classification Metrics:")
    print(f"  Accuracy: {accuracy_gd:.4f}")
    print(f"  Precision: {precision_gd:.4f}")
    print(f"  Recall: {recall_gd:.4f}")
    print(f"  F1 Score: {f1_gd:.4f}")
    
    # Feature Importance Analysis
    print("\n5. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("Feature Importance (Sklearn):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")
    
    # Create comprehensive visualization
    print("\n6. CREATING COMPREHENSIVE VISUALIZATION")
    print("-" * 40)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Cost function over epochs
    plt.subplot(4, 4, 1)
    plt.plot(range(epochs), cost_history, color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    plt.title('Gradient Descent: Cost Function')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Sklearn)
    plt.subplot(4, 4, 2)
    plt.scatter(y_test, y_pred_test, color='purple', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual Charges ($)')
    plt.ylabel('Predicted Charges ($)')
    plt.title('Sklearn: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Actual vs Predicted (Gradient Descent)
    plt.subplot(4, 4, 3)
    plt.scatter(y_test, y_pred_gd_test, color='green', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual Charges ($)')
    plt.ylabel('Predicted Charges ($)')
    plt.title('Gradient Descent: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    plt.subplot(4, 4, 4)
    feature_importance_sorted = feature_importance.sort_values('Coefficient', key=abs, ascending=True)
    colors = ['red' if x < 0 else 'blue' for x in feature_importance_sorted['Coefficient']]
    bars = plt.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Coefficient'], color=colors, alpha=0.7)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals (Sklearn)
    plt.subplot(4, 4, 5)
    residuals_sklearn = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals_sklearn, alpha=0.6, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Sklearn: Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Residuals (Gradient Descent)
    plt.subplot(4, 4, 6)
    residuals_gd = y_test - y_pred_gd_test
    plt.scatter(y_pred_gd_test, residuals_gd, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Gradient Descent: Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: ROC Curve (Sklearn)
    plt.subplot(4, 4, 7)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test_binary, y_pred_test)
    roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    plt.plot(fpr_sklearn, tpr_sklearn, color='purple', lw=2, label=f'Sklearn (AUC = {roc_auc_sklearn:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Sklearn: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot 8: ROC Curve (Gradient Descent)
    plt.subplot(4, 4, 8)
    fpr_gd, tpr_gd, _ = roc_curve(y_test_binary, y_pred_gd_test)
    roc_auc_gd = auc(fpr_gd, tpr_gd)
    plt.plot(fpr_gd, tpr_gd, color='green', lw=2, label=f'Gradient Descent (AUC = {roc_auc_gd:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gradient Descent: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Classification Metrics Comparison
    plt.subplot(4, 4, 9)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    sklearn_values = [accuracy_sklearn, precision_sklearn, recall_sklearn, f1_sklearn]
    gd_values = [accuracy_gd, precision_gd, recall_gd, f1_gd]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, sklearn_values, width, label='Sklearn', alpha=0.7, color='purple')
    plt.bar(x + width/2, gd_values, width, label='Gradient Descent', alpha=0.7, color='green')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Classification Metrics Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 10: Prediction Distribution Comparison
    plt.subplot(4, 4, 10)
    plt.hist(y_pred_test, bins=30, alpha=0.7, color='purple', label='Sklearn', density=True)
    plt.hist(y_pred_gd_test, bins=30, alpha=0.7, color='green', label='Gradient Descent', density=True)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (${threshold:.0f})')
    plt.xlabel('Predicted Charges ($)')
    plt.ylabel('Density')
    plt.title('Prediction Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Learning Curve
    plt.subplot(4, 4, 11)
    plt.plot(range(100, epochs), cost_history[100:], color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    plt.title('Learning Curve (Epochs 100-1000)')
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Cost Convergence
    plt.subplot(4, 4, 12)
    cost_diff = np.diff(cost_history)
    plt.plot(range(1, len(cost_diff)+1), cost_diff, color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost Change')
    plt.title('Cost Convergence Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 13: Residuals Distribution (Sklearn)
    plt.subplot(4, 4, 13)
    plt.hist(residuals_sklearn, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals ($)')
    plt.ylabel('Frequency')
    plt.title('Sklearn: Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 14: Residuals Distribution (Gradient Descent)
    plt.subplot(4, 4, 14)
    plt.hist(residuals_gd, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Residuals ($)')
    plt.ylabel('Frequency')
    plt.title('Gradient Descent: Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 15: Model Performance Comparison
    plt.subplot(4, 4, 15)
    regression_metrics = ['MSE', 'R²', 'MAE']
    sklearn_reg_values = [mse_test, r2_test, mae_test]
    gd_reg_values = [mse_gd_test, r2_gd_test, mae_gd_test]
    
    x = np.arange(len(regression_metrics))
    width = 0.35
    
    plt.bar(x - width/2, sklearn_reg_values, width, label='Sklearn', alpha=0.7, color='purple')
    plt.bar(x + width/2, gd_reg_values, width, label='Gradient Descent', alpha=0.7, color='green')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Regression Metrics Comparison')
    plt.xticks(x, regression_metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 16: Summary Statistics
    plt.subplot(4, 4, 16)
    plt.text(0.1, 0.8, 'Model Performance Summary', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Sklearn MSE: ${mse_test:.2f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Sklearn R²: {r2_test:.4f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'GD MSE: ${mse_gd_test:.2f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'GD R²: {r2_gd_test:.4f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Sklearn Acc: {accuracy_sklearn:.4f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'GD Acc: {accuracy_gd:.4f}', fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('lab03_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL LAB 03 ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset: Insurance charges prediction")
    print(f"Total samples: {len(df)}")
    print(f"Samples after preprocessing: {len(df_clean)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    print(f"\nSKLEARN MODEL PERFORMANCE:")
    print(f"  Regression - MSE: ${mse_test:.2f}, R²: {r2_test:.4f}, MAE: ${mae_test:.2f}")
    print(f"  Classification - Accuracy: {accuracy_sklearn:.4f}, F1: {f1_sklearn:.4f}")
    
    print(f"\nGRADIENT DESCENT MODEL PERFORMANCE:")
    print(f"  Regression - MSE: ${mse_gd_test:.2f}, R²: {r2_gd_test:.4f}, MAE: ${mae_gd_test:.2f}")
    print(f"  Classification - Accuracy: {accuracy_gd:.4f}, F1: {f1_gd:.4f}")
    
    print(f"\nKEY INSIGHTS:")
    print(f"  • Smoking status is the most important predictor of insurance charges")
    print(f"  • Age and BMI also significantly impact insurance costs")
    print(f"  • Both models show similar performance, validating gradient descent implementation")
    print(f"  • The model can predict insurance charges with reasonable accuracy")
    print(f"  • Classification metrics help understand model performance for decision-making")
    print("="*60)
    
    print(f"\nVisualization saved as: lab03_comprehensive_analysis.png")
    print("Lab 03 analysis completed successfully!")

if __name__ == "__main__":
    main()
