# Quiz 1: Python Libraries Setup
# Import required libraries: NumPy, SciPy, Pandas, and Matplotlib

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# Create a 3x3 NumPy array with random integers from 0 to 10 (inclusive)
print("Creating 3x3 NumPy array with random integers (0-10):")
random_array = np.random.randint(0, 11, size=(3, 3))
print(random_array)

# Convert the NumPy array to a Pandas DataFrame
print("\nConverting NumPy array to Pandas DataFrame:")
df = pd.DataFrame(random_array)
print(df)

# Compute and print mean and standard deviation of the DataFrame
print("\nStatistical computations:")
mean_value = df.mean().mean()  # Overall mean of all values
std_value = df.std().mean()    # Overall standard deviation of all values
print(f"Mean of DataFrame: {mean_value:.4f}")
print(f"Standard deviation of DataFrame: {std_value:.4f}")

# Alternative: Mean and std for each column
print("\nMean and standard deviation for each column:")
for i, col in enumerate(df.columns):
    print(f"Column {i}: Mean = {df[col].mean():.4f}, Std = {df[col].std():.4f}")

# Use SciPy to compute eigenvalues of the original NumPy array
print("\nComputing eigenvalues using SciPy:")
eigenvalues = scipy.linalg.eigvals(random_array)
print(f"Eigenvalues: {eigenvalues}")

# Plot histogram of DataFrame values using Matplotlib
print("\nCreating histogram of DataFrame values:")
plt.figure(figsize=(10, 6))
plt.hist(df.values.flatten(), bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of DataFrame Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

print("\nQuiz 1 completed successfully!")
