# Quiz 5: Logistic Regression Classification
# Fit LogisticRegression model and evaluate performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve, auc)

# Create synthetic Social Network Ads dataset
print("Creating Social Network Ads dataset...")
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Generate Age (18-65)
age = np.random.uniform(18, 65, n_samples)

# Generate Estimated Salary (20k-150k)
salary = np.random.uniform(20000, 150000, n_samples)

# Generate Purchased (binary) based on age and salary with some noise
# Higher age and salary increase probability of purchase
purchase_prob = 1 / (1 + np.exp(-(age * 0.05 + salary * 0.00001 - 3)))
purchased = np.random.binomial(1, purchase_prob, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'EstimatedSalary': salary,
    'Purchased': purchased
})

print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")
print(f"\nPurchased distribution:\n{df['Purchased'].value_counts()}")

# Load 'Age' and 'Estimated Salary' as features, 'Purchased' as label
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features using StandardScaler
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled features shape: {X_scaled.shape}")
print(f"Scaled features mean: {X_scaled.mean(axis=0)}")
print(f"Scaled features std: {X_scaled.std(axis=0)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Fit LogisticRegression model
print("\nFitting Logistic Regression model...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Print confusion matrix, accuracy, precision, recall, and F1-score
print("\n=== MODEL EVALUATION ===")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot ROC curve with AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15, 5))

# ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Confusion Matrix Heatmap
plt.subplot(1, 3, 2)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Purchased', 'Purchased'])
plt.yticks(tick_marks, ['Not Purchased', 'Purchased'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

# Decision boundary scatter plot for training set
plt.subplot(1, 3, 3)
# Create a mesh grid
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh grid
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.title('Decision Boundary - Training Set')

plt.tight_layout()
plt.show()

# Additional plot: Decision boundary for test set
plt.figure(figsize=(12, 5))

# Decision boundary for test set
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.title('Decision Boundary - Test Set')

# Feature importance
plt.subplot(1, 2, 2)
feature_names = ['Age', 'Estimated Salary']
coefficients = log_reg.coef_[0]
plt.bar(feature_names, coefficients, color=['skyblue', 'lightcoral'])
plt.title('Feature Importance (Coefficients)')
plt.ylabel('Coefficient Value')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, coef in enumerate(coefficients):
    plt.text(i, coef + 0.01, f'{coef:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\nModel Coefficients:")
print(f"Age coefficient: {log_reg.coef_[0][0]:.4f}")
print(f"Estimated Salary coefficient: {log_reg.coef_[0][1]:.4f}")
print(f"Intercept: {log_reg.intercept_[0]:.4f}")

print("\nQuiz 5 completed successfully!")
