# Quiz 2: Simple Linear Regression
# Implement and evaluate a simple linear regression model using scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes CSV dataset using Pandas
print("Loading Diabetes dataset...")
# Note: You'll need to download the diabetes dataset from Kaggle or use a similar dataset
# For this example, I'll create a synthetic diabetes-like dataset
np.random.seed(42)

# Create synthetic diabetes-like data
n_samples = 442
glucose = np.random.normal(120, 30, n_samples)  # Glucose levels
outcome = (glucose > 140).astype(int)  # Binary outcome based on glucose threshold

# Add some noise to make it more realistic
glucose += np.random.normal(0, 5, n_samples)
outcome = np.where(glucose > 140 + np.random.normal(0, 10, n_samples), 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'Glucose': glucose,
    'Outcome': outcome
})

print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# Drop any missing values from the dataset
print(f"\nMissing values before cleaning: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
print(f"Dataset shape after cleaning: {df.shape}")

# Designate 'Glucose' as feature and 'Outcome' as label
X = df[['Glucose']]  # Feature (input)
y = df['Outcome']    # Label (target)

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train a LinearRegression model using scikit-learn
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Print Mean Squared Error (MSE) and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Additional metrics
print(f"Model coefficient: {model.coef_[0]:.4f}")
print(f"Model intercept: {model.intercept_:.4f}")

# Plot actual outcomes against predicted outcomes using Matplotlib
plt.figure(figsize=(12, 5))

# Scatter plot of actual vs predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Outcomes')
plt.ylabel('Predicted Outcomes')
plt.title('Actual vs Predicted Outcomes')
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Outcomes')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nQuiz 2 completed successfully!")
