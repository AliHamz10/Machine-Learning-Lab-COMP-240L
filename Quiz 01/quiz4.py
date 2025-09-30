# Quiz 4: Multivariate Regression with Feature Scaling
# Build Ridge regression model using scikit-learn Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create synthetic Student Performance dataset
print("Creating Student Performance dataset...")
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Generate features
math_score = np.random.normal(75, 15, n_samples)
reading_score = np.random.normal(80, 12, n_samples)
writing_score = np.random.normal(78, 14, n_samples)
study_hours = np.random.normal(3, 1.5, n_samples)
attendance = np.random.uniform(70, 100, n_samples)

# Generate extracurricular activities (categorical)
activities = np.random.choice(['Sports', 'Music', 'Art', 'None'], n_samples, p=[0.3, 0.25, 0.25, 0.2])

# Generate target variable (GPA) based on features
gpa = (math_score * 0.3 + reading_score * 0.25 + writing_score * 0.25 + 
       study_hours * 2 + attendance * 0.01 + np.random.normal(0, 0.5, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'Math_Score': math_score,
    'Reading_Score': reading_score,
    'Writing_Score': writing_score,
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Extracurricular_Activities': activities,
    'GPA': gpa
})

print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# Encode the 'Extracurricular Activities' column using LabelEncoder
print("\nEncoding 'Extracurricular Activities' column...")
le = LabelEncoder()
df['Extracurricular_Activities_Encoded'] = le.fit_transform(df['Extracurricular_Activities'])

print("Label encoding mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label}: {i}")

# Prepare features and target
feature_columns = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Study_Hours', 'Attendance', 'Extracurricular_Activities_Encoded']
X = df[feature_columns]
y = df['GPA']

print(f"\nFeature columns: {feature_columns}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Build Ridge regression model using Pipeline
print("\nBuilding Ridge regression model with Pipeline...")

# Create pipeline with StandardScaler and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))  # Ridge regression with regularization
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Report MSE and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Get feature coefficients from the Ridge model
ridge_model = pipeline.named_steps['ridge']
coefficients = ridge_model.coef_
feature_names = feature_columns

# Display feature coefficients in a bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(feature_names)), coefficients, color='skyblue', edgecolor='black')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Feature Coefficients')
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, coef) in enumerate(zip(bars, coefficients)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{coef:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Display coefficients in a table
print(f"\nFeature Coefficients:")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.to_string(index=False))

# Additional analysis: Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs Predicted GPA')
plt.grid(True, alpha=0.3)
plt.show()

print("\nQuiz 4 completed successfully!")
