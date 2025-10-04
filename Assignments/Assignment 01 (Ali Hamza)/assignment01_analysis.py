"""
Assignment 01: Machine Learning Analysis
Class: BSAI F23 Red
Student: B23F0063AI106

Business Problem: Wine Quality Prediction for Market Segmentation
Objective: Predict wine quality to help wineries segment their products and optimize pricing strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WineQualityAnalysis:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("="*60)
        print("WINE QUALITY PREDICTION ANALYSIS")
        print("="*60)
        
        # Load wine quality dataset
        try:
            # Try to load from URL first
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            self.df = pd.read_csv(url, sep=';')
            print("✓ Dataset loaded successfully from UCI repository")
        except:
            print("❌ Could not load from URL. Please ensure internet connection.")
            return False
            
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Features: {self.df.shape[1]-1}")
        print(f"Instances: {self.df.shape[0]}")
        
        return True
    
    def explore_data(self):
        """Comprehensive data exploration"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        # Basic info
        print("\n1. DATASET OVERVIEW:")
        print(self.df.info())
        
        print("\n2. FIRST 5 ROWS:")
        print(self.df.head())
        
        print("\n3. STATISTICAL SUMMARY:")
        print(self.df.describe())
        
        # Data types and missing values
        print("\n4. DATA QUALITY:")
        print("Missing values per column:")
        print(self.df.isnull().sum())
        print(f"\nTotal missing values: {self.df.isnull().sum().sum()}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Quality distribution
        print("\n5. TARGET VARIABLE DISTRIBUTION:")
        print(self.df['quality'].value_counts().sort_index())
        
        # Create visualizations
        self.create_exploratory_plots()
        
    def create_exploratory_plots(self):
        """Create comprehensive exploratory visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wine Quality Dataset - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # Quality distribution
        axes[0,0].hist(self.df['quality'], bins=6, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Quality Score Distribution')
        axes[0,0].set_xlabel('Quality Score')
        axes[0,0].set_ylabel('Frequency')
        
        # Correlation heatmap
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Feature Correlation Matrix')
        
        # Alcohol vs Quality
        sns.boxplot(data=self.df, x='quality', y='alcohol', ax=axes[0,2])
        axes[0,2].set_title('Alcohol Content by Quality')
        axes[0,2].set_xlabel('Quality Score')
        axes[0,2].set_ylabel('Alcohol %')
        
        # Volatile acidity vs Quality
        sns.boxplot(data=self.df, x='quality', y='volatile acidity', ax=axes[1,0])
        axes[1,0].set_title('Volatile Acidity by Quality')
        axes[1,0].set_xlabel('Quality Score')
        axes[1,0].set_ylabel('Volatile Acidity')
        
        # Sulphates vs Quality
        sns.boxplot(data=self.df, x='quality', y='sulphates', ax=axes[1,1])
        axes[1,1].set_title('Sulphates by Quality')
        axes[1,1].set_xlabel('Quality Score')
        axes[1,1].set_ylabel('Sulphates')
        
        # Feature importance (using correlation with quality)
        quality_corr = corr_matrix['quality'].drop('quality').abs().sort_values(ascending=True)
        quality_corr.plot(kind='barh', ax=axes[1,2], color='lightcoral')
        axes[1,2].set_title('Feature Correlation with Quality')
        axes[1,2].set_xlabel('Absolute Correlation')
        
        plt.tight_layout()
        plt.savefig('wine_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Create quality categories (business insight: segment wines)
        self.df['quality_category'] = pd.cut(self.df['quality'], 
                                           bins=[0, 5, 6, 10], 
                                           labels=['Low', 'Medium', 'High'])
        
        print("Quality Categories Created:")
        print(self.df['quality_category'].value_counts())
        
        # Feature selection (remove quality, keep quality_category as target)
        feature_cols = [col for col in self.df.columns if col not in ['quality', 'quality_category']]
        X = self.df[feature_cols]
        y = self.df['quality_category']
        
        # Handle any missing values (though dataset is clean)
        X = X.fillna(X.median())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Feature columns: {feature_cols}")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'SVM']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✓ {name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV Mean': f"{metrics['cv_mean']:.4f}",
                'CV Std': f"{metrics['cv_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        
        # Detailed evaluation of best model
        self.detailed_evaluation(best_model_name)
        
    def detailed_evaluation(self, model_name):
        """Detailed evaluation of the best model"""
        print(f"\n" + "="*60)
        print(f"DETAILED EVALUATION: {model_name.upper()}")
        print("="*60)
        
        model = self.models[model_name]
        metrics = self.results[model_name]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, metrics['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, metrics['predictions'])
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importance:")
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def business_insights(self):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\n1. MODEL PERFORMANCE SUMMARY:")
        print(f"   • Best Model: {best_model_name}")
        print(f"   • Accuracy: {best_accuracy:.1%}")
        print(f"   • This model can correctly predict wine quality {best_accuracy:.1%} of the time")
        
        print(f"\n2. BUSINESS APPLICATIONS:")
        print(f"   • Quality Control: Automatically classify wines during production")
        print(f"   • Pricing Strategy: Price wines based on predicted quality tiers")
        print(f"   • Market Segmentation: Target different customer segments")
        print(f"   • Inventory Management: Optimize stock based on quality predictions")
        
        print(f"\n3. KEY FINDINGS:")
        print(f"   • Model achieves {best_accuracy:.1%} accuracy in wine quality prediction")
        print(f"   • This enables data-driven decision making for wine businesses")
        print(f"   • Quality prediction can be automated, reducing manual tasting costs")
        print(f"   • Enables consistent quality standards across production batches")
        
        print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
        print(f"   • Deploy {best_model_name} model in production environment")
        print(f"   • Integrate with existing wine production systems")
        print(f"   • Train staff on interpreting model predictions")
        print(f"   • Monitor model performance and retrain periodically")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Wine Quality Prediction Analysis...")
        
        # Load data
        if not self.load_data():
            return
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Generate business insights
        self.business_insights()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Files generated:")
        print("• wine_quality_analysis.png - Exploratory analysis plots")
        print("• feature_importance_*.png - Feature importance plots")
        print("• This script contains all analysis code")

if __name__ == "__main__":
    # Run the complete analysis
    analysis = WineQualityAnalysis()
    analysis.run_complete_analysis()
