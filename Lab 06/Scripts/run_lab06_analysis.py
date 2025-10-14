#!/usr/bin/env python3
"""
Lab 06: Machine Learning Evaluation on Pulsar Stars Dataset
Comprehensive analysis script for HTRU2 Pulsar Stars classification

Author: Ali Hamza
Course: COMP 240L - Machine Learning
Institution: Pak-Austria Fachhochschule
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   cross_validate, learning_curve, 
                                   cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                           accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, roc_curve,
                           RocCurveDisplay, classification_report,
                           precision_recall_curve, PrecisionRecallDisplay)
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
lab_dir = os.path.dirname(script_dir)
data_path = os.path.join(lab_dir, 'Data', 'pulsar_stars.csv')
results_dir = os.path.join(lab_dir, 'Results')

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

def load_and_explore_data():
    """Load and perform initial data exploration"""
    print("=" * 60)
    print("LAB 06: MACHINE LEARNING EVALUATION ON PULSAR STARS DATASET")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"\n📊 Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    # Basic statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Target distribution
    target_col = 'target_class'
    class_counts = df[target_col].value_counts()
    class_balance = df[target_col].mean()
    
    print(f"\n🎯 Target Distribution:")
    print(f"Non-Pulsars (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"Pulsars (1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")
    print(f"Class balance: {class_balance:.3f}")
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print(f"\n🔧 Data Preprocessing:")
    
    # Handle any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Separate features and target
    target_col = 'target_class'
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target classes: {np.unique(y)}")
    
    return X, y, feature_cols

def perform_statistical_analysis(df, feature_cols):
    """Perform statistical analysis on features"""
    print(f"\n📊 Statistical Analysis:")
    
    target_col = 'target_class'
    results = []
    
    for feature in feature_cols:
        # Separate by class
        class_0 = df[df[target_col] == 0][feature].values
        class_1 = df[df[target_col] == 1][feature].values
        
        # Perform t-test
        t_stat, p_value = ttest_ind(class_0, class_1, equal_var=False)
        
        # Calculate means
        mean_0 = class_0.mean()
        mean_1 = class_1.mean()
        
        results.append({
            'Feature': feature,
            'Mean_Non_Pulsar': mean_0,
            'Mean_Pulsar': mean_1,
            'Difference': mean_1 - mean_0,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        })
    
    stats_df = pd.DataFrame(results)
    print(f"\nT-test Results (Class 0 vs Class 1):")
    print(stats_df.round(4))
    
    significant_features = stats_df[stats_df['Significant']]['Feature'].tolist()
    print(f"\nSignificant features (p < 0.05): {len(significant_features)}/{len(feature_cols)}")
    
    return stats_df

def train_and_evaluate_model(X, y, feature_cols):
    """Train and evaluate the logistic regression model"""
    print(f"\n🤖 Model Training and Evaluation:")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))
    ])
    
    # K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    print(f"\n📊 Cross-Validation Results (5-fold):")
    cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
    
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        print(f"{metric:>9}: mean={vals.mean():.4f}, std={vals.std():.4f}")
    
    # Train on full training set
    pipe.fit(X_train, y_train)
    
    # Test set predictions
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n🎯 Test Set Performance:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    
    return pipe, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_results

def create_visualizations(pipe, X_train, X_test, y_train, y_test, y_pred, y_prob, 
                         cv_results, df, feature_cols, stats_df):
    """Create comprehensive visualizations"""
    print(f"\n📈 Creating Visualizations:")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
    ax.set_title("Confusion Matrix - Pulsar Classification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Pulsar Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Learning Curve - Over/Underfitting Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cross-Validation Results
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    means = [cv_results[f'test_{metric}'].mean() for metric in metrics]
    stds = [cv_results[f'test_{metric}'].std() for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Results (5-fold)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cv_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Feature Importance (Coefficients)
    feature_importance = np.abs(pipe.named_steps['clf'].coef_[0])
    feature_names = feature_cols
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)[::-1]
    bars = ax.barh(range(len(feature_names)), feature_importance[sorted_idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance (|Coefficient|)')
    ax.set_title('Feature Importance - Logistic Regression Coefficients', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(feature_importance[sorted_idx]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Statistical Significance Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    significant = stats_df['Significant']
    colors = ['red' if sig else 'blue' for sig in significant]
    
    bars = ax.barh(range(len(stats_df)), -np.log10(stats_df['P_Value']), color=colors, alpha=0.7)
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05 threshold')
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df['Feature'])
    ax.set_xlabel('-log10(p-value)')
    ax.set_title('Statistical Significance of Features (T-test)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'statistical_significance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(df, pipe, X_test, y_test, y_pred, y_prob, 
                                cv_results, stats_df, feature_cols):
    """Generate comprehensive results report"""
    print(f"\n📝 Generating Comprehensive Report:")
    
    # Calculate final metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Generate detailed results
    results = {
        'Dataset': {
            'Total Samples': len(df),
            'Features': len(feature_cols),
            'Class Distribution': {
                'Non-Pulsars': int(df['target_class'].value_counts()[0]),
                'Pulsars': int(df['target_class'].value_counts()[1]),
                'Class Balance': float(df['target_class'].mean())
            }
        },
        'Model Performance': {
            'Test Accuracy': float(accuracy),
            'Test Precision': float(precision),
            'Test Recall': float(recall),
            'Test F1-Score': float(f1),
            'Test ROC-AUC': float(roc_auc)
        },
        'Cross-Validation': {
            metric: {
                'Mean': float(cv_results[f'test_{metric}'].mean()),
                'Std': float(cv_results[f'test_{metric}'].std())
            } for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        },
        'Statistical Analysis': {
            'Significant Features': int(stats_df['Significant'].sum()),
            'Total Features': len(feature_cols),
            'Significance Rate': float(stats_df['Significant'].mean())
        }
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([
        ['Accuracy', accuracy, cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std()],
        ['Precision', precision, cv_results['test_precision'].mean(), cv_results['test_precision'].std()],
        ['Recall', recall, cv_results['test_recall'].mean(), cv_results['test_recall'].std()],
        ['F1-Score', f1, cv_results['test_f1'].mean(), cv_results['test_f1'].std()],
        ['ROC-AUC', roc_auc, cv_results['test_roc_auc'].mean(), cv_results['test_roc_auc'].std()]
    ], columns=['Metric', 'Test_Score', 'CV_Mean', 'CV_Std'])
    
    results_df.to_csv(os.path.join(results_dir, 'model_results.csv'), index=False)
    
    # Save detailed results
    with open(os.path.join(results_dir, 'detailed_results.txt'), 'w') as f:
        f.write("LAB 06: MACHINE LEARNING EVALUATION ON PULSAR STARS DATASET\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Class Distribution: {df['target_class'].value_counts().to_dict()}\n")
        f.write(f"Class Balance: {df['target_class'].mean():.3f}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n")
        f.write(f"Test ROC-AUC: {roc_auc:.4f}\n\n")
        
        f.write("CROSS-VALIDATION RESULTS:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean_val = cv_results[f'test_{metric}'].mean()
            std_val = cv_results[f'test_{metric}'].std()
            f.write(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write(f"\nSTATISTICAL ANALYSIS:\n")
        f.write(f"Significant Features: {stats_df['Significant'].sum()}/{len(feature_cols)}\n")
        f.write(f"Significance Rate: {stats_df['Significant'].mean():.1%}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    
    print(f"✅ Results saved to {results_dir}/")
    return results

def main():
    """Main execution function"""
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Preprocess data
        X, y, feature_cols = preprocess_data(df)
        
        # Statistical analysis
        stats_df = perform_statistical_analysis(df, feature_cols)
        
        # Train and evaluate model
        pipe, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_results = train_and_evaluate_model(X, y, feature_cols)
        
        # Create visualizations
        create_visualizations(pipe, X_train, X_test, y_train, y_test, y_pred, y_prob, 
                            cv_results, df, feature_cols, stats_df)
        
        # Generate comprehensive report
        results = generate_comprehensive_report(df, pipe, X_test, y_test, y_pred, y_prob, 
                                              cv_results, stats_df, feature_cols)
        
        print(f"\n🎉 Lab 06 Analysis Completed Successfully!")
        print(f"📁 Results saved in: {results_dir}")
        print(f"📊 Generated {len(os.listdir(results_dir))} result files")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
