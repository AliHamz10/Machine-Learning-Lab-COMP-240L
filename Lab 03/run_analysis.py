#!/usr/bin/env python3
"""
Insurance Dataset Linear Regression Analysis
Lab 03: Machine Learning
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    print('=' * 60)
    print('INSURANCE DATASET LINEAR REGRESSION ANALYSIS')
    print('=' * 60)

    # Load the Insurance dataset
    df = pd.read_csv('insurance[1].csv')
    print(f'Dataset Shape: {df.shape}')
    print(f'First 5 rows:')
    print(df.head())

    # Check for missing values
    print(f'Missing Values: {df.isnull().sum().sum()}')

    # Handle categorical variables
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()

    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['smoker_encoded'] = le_smoker.fit_transform(df['smoker'])
    df['region_encoded'] = le_region.fit_transform(df['region'])

    print(f'Encoded values:')
    print(f'Gender: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}')
    print(f'Smoker: {dict(zip(le_smoker.classes_, le_smoker.transform(le_smoker.classes_)))}')
    print(f'Region: {dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))}')

    # Create feature matrix
    feature_cols = ['age', 'gender_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']
    X = df[feature_cols]
    y = df['charges']

    # Outlier analysis
    z_scores = np.abs(stats.zscore(X))
    outlier_count = np.sum((z_scores > 3).any(axis=1))
    df_clean = df[(z_scores < 3).all(axis=1)]
    print(f'Original shape: {df.shape}')
    print(f'Shape after removing outliers: {df_clean.shape}')
    print(f'Outliers removed: {outlier_count} ({outlier_count/df.shape[0]*100:.1f}%)')

    # Prepare clean data
    X_clean = df_clean[feature_cols]
    y_clean = df_clean['charges']

    print('\n' + '=' * 60)
    print('LINEAR REGRESSION MODELS')
    print('=' * 60)

    # Single variable regression
    X_single = X_clean[['age']]
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, y_clean, test_size=0.2, random_state=42
    )
    model_single = LinearRegression()
    model_single.fit(X_train_single, y_train_single)
    y_pred_single = model_single.predict(X_test_single)
    mse_single = mean_squared_error(y_test_single, y_pred_single)
    r2_single = r2_score(y_test_single, y_pred_single)

    print(f'1. Single Variable (Age only):')
    print(f'   MSE: {mse_single:.2f}')
    print(f'   R²:  {r2_single:.4f}')
    print(f'   Coefficient: {model_single.coef_[0]:.2f}')
    print(f'   Intercept: {model_single.intercept_:.2f}')

    # Multi-variable regression
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_multi)
    X_test_scaled = scaler.transform(X_test_multi)
    model_multi = LinearRegression()
    model_multi.fit(X_train_scaled, y_train_multi)
    y_pred_multi = model_multi.predict(X_test_scaled)
    mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
    r2_multi = r2_score(y_test_multi, y_pred_multi)

    print(f'\n2. Multi-Variable (All features):')
    print(f'   MSE: {mse_multi:.2f}')
    print(f'   R²:  {r2_multi:.4f}')
    print(f'   Feature Coefficients:')
    for feature, coef in zip(feature_cols, model_multi.coef_):
        print(f'     {feature}: {coef:.4f}')
    print(f'   Intercept: {model_multi.intercept_:.2f}')

    # Binary classification
    charges_median = y_clean.median()
    y_binary = (y_clean > charges_median).astype(int)
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
        X_clean, y_binary, test_size=0.2, random_state=42
    )
    scaler_binary = StandardScaler()
    X_train_binary_scaled = scaler_binary.fit_transform(X_train_binary)
    X_test_binary_scaled = scaler_binary.transform(X_test_binary)
    model_binary = LinearRegression()
    model_binary.fit(X_train_binary_scaled, y_train_binary)
    y_pred_proba = model_binary.predict(X_test_binary_scaled)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)

    print(f'\n3. Binary Classification (High vs Low Charges):')
    print(f'   Median charges: ${charges_median:.2f}')
    print(f'   Accuracy:  {accuracy:.4f}')
    print(f'   Precision: {precision:.4f}')
    print(f'   Recall:    {recall:.4f}')
    print(f'   F1 Score:  {f1:.4f}')

    # Custom Gradient Descent
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
        return theta, bias, cost_history

    np.random.seed(42)
    m, n = X_train_scaled.shape
    theta = np.random.randn(n)
    bias = 0.0
    learning_rate = 0.01
    epochs = 1000

    theta_optimized, bias_optimized, cost_history = gradient_descent(
        X_train_scaled, y_train_multi, theta, bias, learning_rate, epochs
    )
    y_pred_gd = np.dot(X_test_scaled, theta_optimized) + bias_optimized
    mse_gd = mean_squared_error(y_test_multi, y_pred_gd)
    r2_gd = r2_score(y_test_multi, y_pred_gd)

    print(f'\n4. Custom Gradient Descent:')
    print(f'   MSE: {mse_gd:.2f}')
    print(f'   R²:  {r2_gd:.4f}')
    print(f'   Final cost: {cost_history[-1]:.4f}')

    print('\n' + '=' * 60)
    print('SUMMARY & INSIGHTS')
    print('=' * 60)
    print(f'• Dataset: {df.shape[0]} samples, {df.shape[1]} features')
    print(f'• Data retained after outlier removal: {df_clean.shape[0]} ({df_clean.shape[0]/df.shape[0]*100:.1f}%)')
    print(f'• Most important feature: {feature_cols[np.argmax(np.abs(model_multi.coef_))]} (coef: {model_multi.coef_[np.argmax(np.abs(model_multi.coef_))]:.4f})')
    print(f'• Multi-variable model performs {"better" if r2_multi > r2_single else "worse"} than single variable')
    print(f'• Custom gradient descent {"matches" if abs(r2_gd - r2_multi) < 0.01 else "differs from"} sklearn implementation')
    print('=' * 60)

    # Create visualizations
    create_visualizations(df, df_clean, X_clean, y_clean, feature_cols, 
                         model_single, model_multi, model_binary,
                         y_test_single, y_pred_single, y_test_multi, y_pred_multi,
                         y_test_binary, y_pred_binary, cost_history)

def create_visualizations(df, df_clean, X_clean, y_clean, feature_cols, 
                         model_single, model_multi, model_binary,
                         y_test_single, y_pred_single, y_test_multi, y_pred_multi,
                         y_test_binary, y_pred_binary, cost_history):
    """Create and save visualizations"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_data = X_clean.copy()
    correlation_data['charges'] = y_clean
    sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.3f')
    plt.title('Correlation Matrix - Insurance Dataset')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Distributions
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    df[feature_cols + ['charges']].hist(bins=20, ax=axes[0], color='blue', alpha=0.7)
    axes[0].set_title('Feature Distributions - Before Outlier Removal')
    df_clean[feature_cols + ['charges']].hist(bins=20, ax=axes[1], color='green', alpha=0.7)
    axes[1].set_title('Feature Distributions - After Outlier Removal')
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Single Variable Model Performance
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_single, y_pred_single, color='purple', alpha=0.6)
    plt.plot([y_test_single.min(), y_test_single.max()], 
             [y_test_single.min(), y_test_single.max()], 
             color='red', linewidth=2, linestyle='--')
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Single Variable LR: Actual vs Predicted")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals_single = y_test_single - y_pred_single
    plt.scatter(y_pred_single, residuals_single, color='orange', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('single_variable_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Multi-Variable Model Performance
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_multi, y_pred_multi, color='purple', alpha=0.6)
    plt.plot([y_test_multi.min(), y_test_multi.max()], 
             [y_test_multi.min(), y_test_multi.max()], 
             color='red', linewidth=2, linestyle='--')
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Multi-Variable LR: Actual vs Predicted")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals_multi = y_test_multi - y_pred_multi
    plt.scatter(y_pred_multi, residuals_multi, color='orange', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_variable_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature Importance
    plt.figure(figsize=(12, 6))
    importance = np.abs(model_multi.coef_)
    plt.barh(feature_cols, importance, color='teal', alpha=0.7)
    plt.title('Feature Importance in Linear Regression')
    plt.xlabel('Absolute Coefficient Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Gradient Descent Cost Function
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost Over Epochs (Gradient Descent)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_descent_cost.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations saved:")
    print("- correlation_matrix.png")
    print("- feature_distributions.png") 
    print("- single_variable_model.png")
    print("- multi_variable_model.png")
    print("- feature_importance.png")
    print("- gradient_descent_cost.png")

if __name__ == "__main__":
    main()
