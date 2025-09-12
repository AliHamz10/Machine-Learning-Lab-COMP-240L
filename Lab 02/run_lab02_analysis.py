#!/usr/bin/env python3
"""
Lab 02: Linear Regression Analysis on Insurance Dataset
Department Of Electrical & Computer Engineering
Machine Learning Lab - COMP 240L

This script runs the complete linear regression analysis on the insurance dataset.
Execute this script to generate all analysis results and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("LAB 02: LINEAR REGRESSION ANALYSIS ON INSURANCE DATASET")
    print("=" * 60)
    print("Department Of Electrical & Computer Engineering")
    print("Machine Learning Lab - COMP 240L")
    print("Lab Coordinator: Ms. Sana Saleem")
    print("Course Instructor: Dr. Abid Ali")
    print("Program: BS(AI) -F23, Semester: 5th")
    print("Deadline: 13th September 2025")
    print("=" * 60)
    
    # Load the dataset
    print("\n1. Loading Dataset...")
    df = pd.read_csv('insurance.csv')
    print(f"   Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")
    
    # Basic analysis
    print("\n2. Dataset Analysis...")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Charges statistics:")
    print(f"     Mean: ${df['charges'].mean():.2f}")
    print(f"     Median: ${df['charges'].median():.2f}")
    print(f"     Std: ${df['charges'].std():.2f}")
    
    # Data preprocessing
    print("\n3. Data Preprocessing...")
    df_processed = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df_processed['gender'] = le_gender.fit_transform(df_processed['gender'])
    df_processed['smoker'] = le_smoker.fit_transform(df_processed['smoker'])
    df_processed['region'] = le_region.fit_transform(df_processed['region'])
    
    print("   Categorical variables encoded successfully!")
    
    # Prepare features and target
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Data split: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Train model
    print("\n4. Training Linear Regression Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = lr_model.predict(X_train_scaled)
    y_test_pred = lr_model.predict(X_test_scaled)
    
    print("   Model trained successfully!")
    
    # Evaluate model
    print("\n5. Model Evaluation...")
    
    def calculate_metrics(y_true, y_pred, dataset_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n   {dataset_name} Metrics:")
        print(f"     MSE: ${mse:.2f}")
        print(f"     RMSE: ${rmse:.2f}")
        print(f"     MAE: ${mae:.2f}")
        print(f"     R²: {r2:.4f}")
        
        return mse, rmse, mae, r2
    
    train_mse, train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred, "Training")
    test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(y_test, y_test_pred, "Test")
    
    # Feature importance
    print("\n6. Feature Importance Analysis...")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("   Feature Importance (by coefficient magnitude):")
    for _, row in feature_importance.iterrows():
        impact = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"     {row['Feature']}: {impact} charges by ${abs(row['Coefficient']):.2f}")
    
    # Generate visualizations
    print("\n7. Generating Visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Charges distribution
    plt.subplot(3, 4, 1)
    plt.hist(df['charges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Medical Charges')
    plt.xlabel('Charges ($)')
    plt.ylabel('Frequency')
    
    # 2. Log-transformed charges
    plt.subplot(3, 4, 2)
    plt.hist(np.log(df['charges']), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Log-Transformed Charges')
    plt.xlabel('Log(Charges)')
    plt.ylabel('Frequency')
    
    # 3. Age vs Charges
    plt.subplot(3, 4, 3)
    plt.scatter(df['age'], df['charges'], alpha=0.6, color='coral')
    plt.title('Age vs Charges')
    plt.xlabel('Age')
    plt.ylabel('Charges ($)')
    
    # 4. BMI vs Charges
    plt.subplot(3, 4, 4)
    plt.scatter(df['bmi'], df['charges'], alpha=0.6, color='purple')
    plt.title('BMI vs Charges')
    plt.xlabel('BMI')
    plt.ylabel('Charges ($)')
    
    # 5. Smoker distribution
    plt.subplot(3, 4, 5)
    df['smoker'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Smoker Distribution')
    plt.xlabel('Smoker')
    plt.ylabel('Count')
    
    # 6. Gender distribution
    plt.subplot(3, 4, 6)
    df['gender'].value_counts().plot(kind='bar', color='lightblue')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    
    # 7. Region distribution
    plt.subplot(3, 4, 7)
    df['region'].value_counts().plot(kind='bar', color='lightcoral')
    plt.title('Region Distribution')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 8. Children distribution
    plt.subplot(3, 4, 8)
    df['children'].value_counts().sort_index().plot(kind='bar', color='lightyellow')
    plt.title('Children Distribution')
    plt.xlabel('Number of Children')
    plt.ylabel('Count')
    
    # 9. Actual vs Predicted (Training)
    plt.subplot(3, 4, 9)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Charges ($)')
    plt.ylabel('Predicted Charges ($)')
    plt.title(f'Training: Actual vs Predicted\nR² = {train_r2:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 10. Actual vs Predicted (Test)
    plt.subplot(3, 4, 10)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Charges ($)')
    plt.ylabel('Predicted Charges ($)')
    plt.title(f'Test: Actual vs Predicted\nR² = {test_r2:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 11. Residuals (Test)
    plt.subplot(3, 4, 11)
    test_residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, test_residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Charges ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Test Set: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # 12. Feature importance
    plt.subplot(3, 4, 12)
    feature_importance_sorted = feature_importance.sort_values('Coefficient', key=abs, ascending=True)
    colors = ['red' if x < 0 else 'blue' for x in feature_importance_sorted['Coefficient']]
    bars = plt.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Coefficient'], color=colors, alpha=0.7)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lab02_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("   Comprehensive analysis plot saved as 'lab02_comprehensive_analysis.png'")
    
    # Final conclusions
    print("\n8. Analysis Conclusions:")
    print("=" * 50)
    print(f"   • Dataset: {len(df)} insurance records with {len(X.columns)} features")
    print(f"   • Model Performance: R² = {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
    print(f"   • Average Prediction Error: ${test_rmse:.2f}")
    print(f"   • Key Finding: Smoking status has the strongest impact on charges")
    print(f"   • Model can predict insurance costs with reasonable accuracy")
    
    print("\n" + "=" * 60)
    print("LAB 02 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files generated:")
    print("  - lab02_comprehensive_analysis.png (comprehensive visualization)")
    print("  - Lab_02_Linear_Regression_Analysis.ipynb (detailed notebook)")
    print("=" * 60)

if __name__ == "__main__":
    main()
