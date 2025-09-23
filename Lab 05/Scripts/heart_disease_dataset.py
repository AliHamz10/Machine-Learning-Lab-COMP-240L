#!/usr/bin/env python3
"""
Heart Disease Dataset Generator
Creates a synthetic heart disease dataset for Lab 05 analysis
"""

import pandas as pd
import numpy as np

def generate_heart_disease_dataset(n_samples=1000, random_state=42):
    """
    Generate synthetic heart disease dataset based on real-world patterns
    
    Parameters:
    n_samples (int): Number of samples to generate
    random_state (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate features with realistic distributions
    age = np.random.normal(55, 10, n_samples).astype(int)
    age = np.clip(age, 29, 77)  # Realistic age range
    
    sex = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])  # 0 = female, 1 = male
    
    # Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.3, 0.4])
    
    # Resting blood pressure (mmHg)
    trestbps = np.random.normal(130, 20, n_samples).astype(int)
    trestbps = np.clip(trestbps, 94, 200)
    
    # Serum cholesterol (mg/dl)
    chol = np.random.normal(250, 50, n_samples).astype(int)
    chol = np.clip(chol, 126, 564)
    
    # Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    # Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])
    
    # Maximum heart rate achieved
    thalach = np.random.normal(150, 25, n_samples).astype(int)
    thalach = np.clip(thalach, 71, 202)
    
    # Exercise induced angina (0 = no, 1 = yes)
    exang = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    
    # ST depression induced by exercise relative to rest
    oldpeak = np.random.exponential(1, n_samples)
    oldpeak = np.clip(oldpeak, 0, 6.2)
    
    # Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
    
    # Number of major vessels colored by flourosopy (0-3)
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Thalassemia (0: normal, 1: fixed defect, 2: reversable defect)
    thal = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])
    
    # Create target variable with logical relationships
    heart_disease = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.05  # base probability
        
        # Age factor
        if age[i] > 60:
            prob += 0.25
        elif age[i] > 50:
            prob += 0.15
        
        # Gender factor (males have higher risk)
        if sex[i] == 1:
            prob += 0.1
        
        # Chest pain factor
        if cp[i] == 0:  # typical angina
            prob += 0.3
        elif cp[i] == 1:  # atypical angina
            prob += 0.2
        elif cp[i] == 2:  # non-anginal pain
            prob += 0.1
        
        # Blood pressure factor
        if trestbps[i] > 140:
            prob += 0.15
        elif trestbps[i] > 130:
            prob += 0.1
        
        # Cholesterol factor
        if chol[i] > 300:
            prob += 0.2
        elif chol[i] > 250:
            prob += 0.1
        
        # Heart rate factor
        if thalach[i] < 120:
            prob += 0.2
        elif thalach[i] < 140:
            prob += 0.1
        
        # Exercise factors
        if exang[i] == 1:
            prob += 0.15
        
        if oldpeak[i] > 2:
            prob += 0.2
        elif oldpeak[i] > 1:
            prob += 0.1
        
        # Slope factor
        if slope[i] == 2:  # downsloping
            prob += 0.15
        elif slope[i] == 1:  # flat
            prob += 0.05
        
        # Vessel factor
        if ca[i] > 0:
            prob += 0.1 * ca[i]
        
        # Thalassemia factor
        if thal[i] == 1:  # fixed defect
            prob += 0.2
        elif thal[i] == 2:  # reversable defect
            prob += 0.15
        
        # Cap probability between 0 and 1
        prob = min(prob, 0.95)
        
        heart_disease[i] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': heart_disease
    }
    
    df = pd.DataFrame(data)
    
    # Add feature descriptions
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (0 = female, 1 = male)',
        'cp': 'Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)',
        'trestbps': 'Resting blood pressure (mmHg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (0=false, 1=true)',
        'restecg': 'Resting electrocardiographic results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (0=no, 1=yes)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)',
        'ca': 'Number of major vessels colored by flourosopy (0-3)',
        'thal': 'Thalassemia (0=normal, 1=fixed defect, 2=reversable defect)',
        'target': 'Heart disease (0=no disease, 1=disease)'
    }
    
    return df, feature_descriptions

def main():
    """Generate and save the dataset"""
    print("Generating Heart Disease Dataset...")
    
    # Generate dataset
    df, descriptions = generate_heart_disease_dataset(n_samples=1000, random_state=42)
    
    # Save dataset
    df.to_csv('heart_disease_dataset.csv', index=False)
    
    # Save feature descriptions
    with open('feature_descriptions.txt', 'w') as f:
        f.write("Heart Disease Dataset - Feature Descriptions\n")
        f.write("=" * 50 + "\n\n")
        for feature, description in descriptions.items():
            f.write(f"{feature}: {description}\n")
    
    print(f"✓ Dataset generated successfully!")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns)-1}")
    print(f"  - Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"  - Files created: heart_disease_dataset.csv, feature_descriptions.txt")
    
    return df

if __name__ == "__main__":
    main()
