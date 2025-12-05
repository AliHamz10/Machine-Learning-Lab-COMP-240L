"""
CCP: Comparative Machine Learning Framework for Medical Diagnostic Prediction
Course: COMP-240L Machine Learning Lab
Student: Ali Hamza
Dataset: Cleveland Heart Disease Dataset

Objective: Develop and critically evaluate a predictive diagnostic system using
multiple machine learning algorithms to diagnose heart disease from patient clinical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score, roc_auc_score,
                           roc_curve, precision_recall_curve)
from imbalanced-learn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartDiseaseMLFramework:
    """
    Comprehensive ML framework for heart disease prediction using multiple algorithms.
    """
    
    def __init__(self, results_dir='../Results'):
        """Initialize the framework"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.tuned_models = {}
        self.results = {}
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Column names for Cleveland Heart Disease dataset
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
    def load_data(self):
        """Load Cleveland Heart Disease dataset from UCI ML Repository"""
        print("="*70)
        print("HEART DISEASE PREDICTION - COMPARATIVE ML FRAMEWORK")
        print("="*70)
        
        # Try multiple sources for the dataset
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
        ]
        
        # Try loading from UCI first
        try:
            self.df = pd.read_csv(urls[0], names=self.column_names, na_values='?')
            print("✓ Dataset loaded successfully from UCI ML Repository")
        except:
            try:
                # Try alternative source
                self.df = pd.read_csv(urls[1])
                if 'target' not in self.df.columns:
                    # If using alternative source, ensure target column exists
                    if 'target' in self.df.columns:
                        pass
                    else:
                        raise ValueError("Target column not found")
                print("✓ Dataset loaded successfully from alternative source")
            except:
                # Try loading from local file
                try:
                    data_path = '../Data/heart_disease.csv'
                    self.df = pd.read_csv(data_path, names=self.column_names, na_values='?')
                    print("✓ Dataset loaded from local file")
                except:
                    print("❌ Could not load dataset. Please ensure internet connection or local file exists.")
                    print("   Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
                    return False
        
        # Handle target column - ensure binary classification (0 = no disease, 1 = disease)
        if 'target' in self.df.columns:
            # Convert to binary: 0 = no disease, 1-4 = disease
            self.df['target'] = (self.df['target'] > 0).astype(int)
        else:
            print("❌ Target column not found in dataset")
            return False
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Features: {self.df.shape[1]-1}")
        print(f"Instances: {self.df.shape[0]}")
        
        return True
    
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*70)
        
        # Basic info
        print("\n1. DATASET OVERVIEW:")
        print(self.df.info())
        
        print("\n2. FIRST 5 ROWS:")
        print(self.df.head())
        
        print("\n3. STATISTICAL SUMMARY:")
        print(self.df.describe())
        
        # Data quality check
        print("\n4. DATA QUALITY ANALYSIS:")
        print("Missing values per column:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        print(f"\nTotal missing values: {self.df.isnull().sum().sum()}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Target distribution
        print("\n5. TARGET VARIABLE DISTRIBUTION:")
        target_dist = self.df['target'].value_counts()
        print(target_dist)
        print(f"\nClass 0 (No Disease): {target_dist[0]} ({target_dist[0]/len(self.df)*100:.2f}%)")
        print(f"Class 1 (Disease): {target_dist[1]} ({target_dist[1]/len(self.df)*100:.2f}%)")
        
        # Check for class imbalance
        imbalance_ratio = min(target_dist) / max(target_dist)
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.3f}")
        if imbalance_ratio < 0.5:
            print("⚠ Warning: Significant class imbalance detected. Will use appropriate techniques.")
        
        # Create visualizations
        self.create_exploratory_plots()
        
    def create_exploratory_plots(self):
        """Create comprehensive exploratory visualizations"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Heart Disease Dataset - Exploratory Data Analysis', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Target distribution
        ax1 = fig.add_subplot(gs[0, 0])
        target_counts = self.df['target'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        ax1.bar(['No Disease', 'Disease'], target_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Target Variable Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        for i, v in enumerate(target_counts.values):
            ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # 2. Correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1:])
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=ax2, cbar_kws={'shrink': 0.8}, fmt='.2f', square=True)
        ax2.set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 3. Age distribution by target
        ax3 = fig.add_subplot(gs[1, 0])
        self.df.boxplot(column='age', by='target', ax=ax3)
        ax3.set_title('Age Distribution by Disease Status', fontweight='bold')
        ax3.set_xlabel('Disease Status')
        ax3.set_ylabel('Age')
        plt.setp(ax3.get_xticklabels(), rotation=0)
        
        # 4. Cholesterol distribution by target
        ax4 = fig.add_subplot(gs[1, 1])
        self.df.boxplot(column='chol', by='target', ax=ax4)
        ax4.set_title('Cholesterol Distribution by Disease Status', fontweight='bold')
        ax4.set_xlabel('Disease Status')
        ax4.set_ylabel('Cholesterol (mg/dl)')
        plt.setp(ax4.get_xticklabels(), rotation=0)
        
        # 5. Maximum heart rate by target
        ax5 = fig.add_subplot(gs[1, 2])
        self.df.boxplot(column='thalach', by='target', ax=ax5)
        ax5.set_title('Max Heart Rate by Disease Status', fontweight='bold')
        ax5.set_xlabel('Disease Status')
        ax5.set_ylabel('Max Heart Rate')
        plt.setp(ax5.get_xticklabels(), rotation=0)
        
        # 6. Feature correlation with target
        ax6 = fig.add_subplot(gs[2, 0])
        target_corr = corr_matrix['target'].drop('target').abs().sort_values(ascending=True)
        target_corr.plot(kind='barh', ax=ax6, color='lightcoral')
        ax6.set_title('Feature Correlation with Target', fontweight='bold')
        ax6.set_xlabel('Absolute Correlation')
        
        # 7. Sex distribution by target
        ax7 = fig.add_subplot(gs[2, 1])
        sex_target = pd.crosstab(self.df['sex'], self.df['target'])
        sex_target.plot(kind='bar', ax=ax7, color=['#ff9999', '#66b3ff'], alpha=0.7)
        ax7.set_title('Disease Status by Sex', fontweight='bold')
        ax7.set_xlabel('Sex (0=Female, 1=Male)')
        ax7.set_ylabel('Count')
        ax7.legend(['No Disease', 'Disease'])
        ax7.set_xticklabels(['Female', 'Male'], rotation=0)
        
        # 8. Chest pain type by target
        ax8 = fig.add_subplot(gs[2, 2])
        cp_target = pd.crosstab(self.df['cp'], self.df['target'])
        cp_target.plot(kind='bar', ax=ax8, color=['#ff9999', '#66b3ff'], alpha=0.7)
        ax8.set_title('Disease Status by Chest Pain Type', fontweight='bold')
        ax8.set_xlabel('Chest Pain Type')
        ax8.set_ylabel('Count')
        ax8.legend(['No Disease', 'Disease'])
        ax8.set_xticklabels(['Typical Angina', 'Atypical Angina', 
                            'Non-anginal Pain', 'Asymptomatic'], rotation=45, ha='right')
        
        plt.savefig(os.path.join(self.results_dir, 'heart_disease_eda.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ EDA visualizations saved to {self.results_dir}/heart_disease_eda.png")
        plt.close()
        
    def preprocess_data(self):
        """Data preprocessing: handle missing values, encoding, scaling, and splitting"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING")
        print("="*70)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling Missing Values:")
        missing_before = df_processed.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        
        # Replace missing values with median for numerical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"   ✓ Filled {col} missing values with median: {median_val:.2f}")
        
        missing_after = df_processed.isnull().sum().sum()
        print(f"   Missing values after: {missing_after}")
        
        # 2. Separate features and target
        X = df_processed.drop('target', axis=1)
        y = df_processed['target']
        
        print(f"\n2. Feature and Target Separation:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        # 3. Train-test split with stratification
        print("\n3. Train-Test Split:")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        print(f"   Training class distribution:\n{self.y_train.value_counts().to_dict()}")
        print(f"   Test class distribution:\n{self.y_test.value_counts().to_dict()}")
        
        # 4. Feature scaling
        print("\n4. Feature Scaling:")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("   ✓ Features scaled using StandardScaler")
        
        # 5. Handle class imbalance using SMOTE
        print("\n5. Handling Class Imbalance:")
        imbalance_ratio = min(self.y_train.value_counts()) / max(self.y_train.value_counts())
        print(f"   Class imbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.6:
            print("   Applying SMOTE to balance training data...")
            smote = SMOTE(random_state=42)
            self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)
            print(f"   ✓ Training data after SMOTE: {self.X_train_scaled.shape[0]} samples")
            print(f"   Balanced class distribution:\n{pd.Series(self.y_train).value_counts().to_dict()}")
        else:
            print("   Class imbalance is acceptable, skipping SMOTE")
        
        print("\n✓ Data preprocessing completed successfully!")
        
    def train_models(self):
        """Train all machine learning models"""
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        # Define all models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True, kernel='rbf'),
            'ANN': MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50), 
                                max_iter=500, early_stopping=True),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Training {name}...")
            print(f"{'='*70}")
            
            # Select appropriate data (scaled for SVM, ANN, KNN)
            if name in ['SVM', 'ANN', 'KNN']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train.values
                X_test_use = self.X_test.values
            
            # Train model
            try:
                model.fit(X_train_use, self.y_train)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC-ROC
            auc_roc = None
            if y_pred_proba is not None:
                try:
                    auc_roc = roc_auc_score(self.y_test, y_pred_proba)
                except:
                    pass
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store model and results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if auc_roc:
                print(f"  AUC-ROC: {auc_roc:.4f}")
            print(f"  CV Score (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}")
        
        print("\n✓ All models trained successfully!")
        
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for Random Forest and SVM"""
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING")
        print("="*70)
        
        # 1. Random Forest Tuning
        print("\n1. Tuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_base = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='accuracy', 
                              n_jobs=-1, verbose=1)
        rf_grid.fit(self.X_train.values, self.y_train)
        
        print(f"   Best parameters: {rf_grid.best_params_}")
        print(f"   Best CV score: {rf_grid.best_score_:.4f}")
        
        # Evaluate tuned model
        rf_tuned = rf_grid.best_estimator_
        rf_tuned_pred = rf_tuned.predict(self.X_test.values)
        rf_tuned_acc = accuracy_score(self.y_test, rf_tuned_pred)
        print(f"   Test accuracy: {rf_tuned_acc:.4f}")
        
        self.tuned_models['Random Forest'] = rf_tuned
        self.models['Random Forest'] = rf_tuned  # Update with tuned model
        
        # 2. SVM Tuning
        print("\n2. Tuning SVM...")
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        
        svm_base = SVC(random_state=42, probability=True)
        svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=5, scoring='accuracy',
                               n_jobs=-1, verbose=1)
        svm_grid.fit(self.X_train_scaled, self.y_train)
        
        print(f"   Best parameters: {svm_grid.best_params_}")
        print(f"   Best CV score: {svm_grid.best_score_:.4f}")
        
        # Evaluate tuned model
        svm_tuned = svm_grid.best_estimator_
        svm_tuned_pred = svm_tuned.predict(self.X_test_scaled)
        svm_tuned_acc = accuracy_score(self.y_test, svm_tuned_pred)
        print(f"   Test accuracy: {svm_tuned_acc:.4f}")
        
        self.tuned_models['SVM'] = svm_tuned
        self.models['SVM'] = svm_tuned  # Update with tuned model
        
        # Update results for tuned models
        for name in ['Random Forest', 'SVM']:
            model = self.models[name]
            if name == 'SVM':
                X_test_use = self.X_test_scaled
            else:
                X_test_use = self.X_test.values
            
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        print("\n✓ Hyperparameter tuning completed!")
        
    def evaluate_models(self):
        """Comprehensive model evaluation and comparison"""
        print("\n" + "="*70)
        print("MODEL EVALUATION & COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics.get('auc_roc', np.nan),
                'CV Mean': metrics.get('cv_mean', np.nan),
                'CV Std': metrics.get('cv_std', np.nan)
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Comparison Summary:")
        print(self.comparison_df.to_string(index=False))
        
        # Save comparison to CSV
        csv_path = os.path.join(self.results_dir, 'model_comparison.csv')
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison results saved to {csv_path}")
        
        # Identify best model
        best_model_name = self.comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {self.comparison_df.iloc[0]['Accuracy']:.4f}")
        print(f"   F1-Score: {self.comparison_df.iloc[0]['F1-Score']:.4f}")
        
        return best_model_name
    
    def create_visualizations(self):
        """Create comprehensive visualizations for model comparison"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # 1. ROC Curves
        self.plot_roc_curves()
        
        # 2. Confusion Matrices
        self.plot_confusion_matrices()
        
        # 3. Model Comparison Charts
        self.plot_model_comparison()
        
        # 4. Feature Importance
        self.plot_feature_importance()
        
        # 5. Learning Curves
        self.plot_learning_curves()
        
        # 6. Radar Chart
        self.plot_radar_chart()
        
        print("\n✓ All visualizations created successfully!")
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.results.items():
            if metrics.get('y_pred_proba') is not None:
                fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
                auc = metrics.get('auc_roc', 0)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved")
        plt.close()
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot normalized confusion matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       ax=axes[idx], cbar_kws={'shrink': 0.8})
            axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}', 
                              fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['No Disease', 'Disease'])
            axes[idx].set_yticklabels(['No Disease', 'Disease'])
        
        # Hide unused subplot
        if n_models < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved")
        plt.close()
        
    def plot_model_comparison(self):
        """Plot bar charts comparing model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.comparison_df)))
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            bars = ax.barh(self.comparison_df['Model'], self.comparison_df[metric], 
                          color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Model Comparison - {metric}', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, self.comparison_df[metric])):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Model comparison charts saved")
        plt.close()
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = ['Decision Tree', 'Random Forest']
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, name in enumerate(tree_models):
            if name in self.models:
                model = self.models[name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.X_train.columns
                    
                    # Sort by importance
                    indices = np.argsort(importances)[::-1]
                    
                    ax = axes[idx]
                    ax.barh(range(len(importances)), importances[indices], 
                           color='steelblue', alpha=0.7, edgecolor='black')
                    ax.set_yticks(range(len(importances)))
                    ax.set_yticklabels([feature_names[i] for i in indices])
                    ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
                    ax.set_title(f'{name} - Feature Importance', fontsize=12, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Feature importance plots saved")
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves for best model to show overfitting/underfitting"""
        # Select best model for learning curve
        best_model_name = self.comparison_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        # Determine which data to use
        if best_model_name in ['SVM', 'ANN', 'KNN']:
            X_use = self.X_train_scaled
        else:
            X_use = self.X_train.values
        
        print(f"\nGenerating learning curve for {best_model_name}...")
        
        # Calculate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_use, self.y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        plt.title(f'Learning Curve - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'learning_curve.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Learning curve saved")
        plt.close()
    
    def plot_radar_chart(self):
        """Create radar/spider chart for multi-metric comparison"""
        
        # Select metrics for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        num_metrics = len(metrics)
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.comparison_df)))
        
        for idx, row in self.comparison_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Radar chart saved")
        plt.close()
        
    def save_models(self):
        """Save trained models"""
        models_dir = os.path.join(self.results_dir, 'saved_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(models_dir, filename)
            joblib.dump(model, filepath)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n✓ Models saved to {models_dir}/")
        
    def generate_summary_report(self):
        """Generate a text summary report"""
        report_path = os.path.join(self.results_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HEART DISEASE PREDICTION - MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"  Total Samples: {len(self.df)}\n")
            f.write(f"  Features: {len(self.df.columns) - 1}\n")
            f.write(f"  Training Samples: {len(self.X_train)}\n")
            f.write(f"  Test Samples: {len(self.X_test)}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write(self.comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("BEST MODEL:\n")
            best_model = self.comparison_df.iloc[0]
            f.write(f"  Model: {best_model['Model']}\n")
            f.write(f"  Accuracy: {best_model['Accuracy']:.4f}\n")
            f.write(f"  F1-Score: {best_model['F1-Score']:.4f}\n")
            f.write(f"  AUC-ROC: {best_model.get('AUC-ROC', 'N/A')}\n")
            
        print(f"✓ Summary report saved to {report_path}")


def main():
    """Main execution function"""
    # Initialize framework
    framework = HeartDiseaseMLFramework(results_dir='../Results')
    
    # Execute pipeline
    if not framework.load_data():
        return
    
    framework.explore_data()
    framework.preprocess_data()
    framework.train_models()
    framework.hyperparameter_tuning()
    best_model = framework.evaluate_models()
    framework.create_visualizations()
    framework.save_models()
    framework.generate_summary_report()
    
    print("\n" + "="*70)
    print("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBest Model: {best_model}")
    print(f"\nAll results saved to: {framework.results_dir}/")
    print("\nGenerated Files:")
    print("  - heart_disease_eda.png")
    print("  - roc_curves.png")
    print("  - confusion_matrices.png")
    print("  - model_comparison.png")
    print("  - feature_importance.png")
    print("  - learning_curve.png")
    print("  - radar_chart.png")
    print("  - model_comparison.csv")
    print("  - summary_report.txt")
    print("  - saved_models/ (directory with all trained models)")


if __name__ == "__main__":
    main()

