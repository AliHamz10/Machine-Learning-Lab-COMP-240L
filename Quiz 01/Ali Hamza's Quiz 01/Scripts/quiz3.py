# Quiz 3: Gradient Descent & Cost Function
# Implement linear regression from scratch using NumPy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the cost function
def compute_cost(X, y, theta, b):
    """
    Compute the cost function for linear regression
    
    Parameters:
    X: features (m x n)
    y: actual labels (m x 1)
    theta: parameters/weights (n x 1)
    b: bias (scalar)
    
    Returns:
    cost: Mean Squared Error
    """
    m = X.shape[0]
    # Compute predictions: h = X * theta + b
    predictions = np.dot(X, theta) + b
    # Compute cost: MSE = (1/2m) * sum((predictions - y)^2)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Define the gradient descent function
def gradient_descent(X, y, theta, b, alpha, epochs):
    """
    Perform gradient descent optimization
    
    Parameters:
    X: features (m x n)
    y: actual labels (m x 1)
    theta: initial parameters/weights (n x 1)
    b: initial bias (scalar)
    alpha: learning rate
    epochs: number of iterations
    
    Returns:
    theta: optimized parameters
    b: optimized bias
    costs: list of costs for each epoch
    """
    m = X.shape[0]
    costs = []
    
    for i in range(epochs):
        # Compute predictions
        predictions = np.dot(X, theta) + b
        
        # Compute gradients
        dtheta = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        # Update parameters
        theta = theta - alpha * dtheta
        b = b - alpha * db
        
        # Compute and store cost
        cost = compute_cost(X, y, theta, b)
        costs.append(cost)
        
        # Print cost every 100 epochs
        if (i + 1) % 100 == 0:
            print(f"Epoch {i+1}: Cost = {cost:.6f}")
    
    return theta, b, costs

# Load and standardize the Diabetes data
print("Loading and preparing Diabetes data...")

# Create synthetic diabetes-like data (similar to quiz2)
np.random.seed(42)
n_samples = 442
glucose = np.random.normal(120, 30, n_samples)
outcome = (glucose > 140).astype(int)
glucose += np.random.normal(0, 5, n_samples)
outcome = np.where(glucose > 140 + np.random.normal(0, 10, n_samples), 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'Glucose': glucose,
    'Outcome': outcome
})

# Prepare features and target
X = df[['Glucose']].values
y = df['Outcome'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")
print(f"Target shape: {y.shape}")

# Initialize parameters
theta = np.random.randn(X_scaled.shape[1], 1)  # Random initial weights
b = 0  # Initial bias
alpha = 0.01  # Learning rate
epochs = 500  # Number of iterations

print(f"\nInitial parameters:")
print(f"Theta: {theta.flatten()}")
print(f"Bias: {b}")

# Run gradient descent for 500 epochs
print(f"\nRunning gradient descent for {epochs} epochs...")
print("Cost every 100 epochs:")
theta_optimized, b_optimized, costs = gradient_descent(X_scaled, y, theta, b, alpha, epochs)

print(f"\nOptimized parameters:")
print(f"Theta: {theta_optimized.flatten()}")
print(f"Bias: {b_optimized}")

# Plot cost vs epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), costs, 'b-', linewidth=2)
plt.title('Cost Function vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.grid(True, alpha=0.3)
plt.show()

# Make predictions with optimized model
predictions = np.dot(X_scaled, theta_optimized) + b_optimized

# Calculate final metrics
final_cost = compute_cost(X_scaled, y, theta_optimized, b_optimized)
print(f"\nFinal cost: {final_cost:.6f}")

# Convert predictions to binary (0 or 1) for classification
predictions_binary = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions_binary == y)
print(f"Accuracy: {accuracy:.4f}")

print("\nQuiz 3 completed successfully!")
