"""
Script to download Cleveland Heart Disease dataset from UCI ML Repository
"""

import pandas as pd
import os

def download_heart_disease_dataset():
    """Download and save Cleveland Heart Disease dataset"""
    
    # Column names for Cleveland Heart Disease dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # UCI ML Repository URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    print("Downloading Cleveland Heart Disease dataset from UCI ML Repository...")
    
    try:
        # Download dataset
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Convert target to binary (0 = no disease, 1-4 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Save to CSV
        output_path = 'heart_disease.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Saved to: {output_path}")
        print(f"\nDataset Info:")
        print(df.info())
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nTarget distribution:")
        print(df['target'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("\nAlternative: You can manually download from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
        return None

if __name__ == "__main__":
    download_heart_disease_dataset()

