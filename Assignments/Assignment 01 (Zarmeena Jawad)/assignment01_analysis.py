"""
Assignment 01: Machine Learning Analysis
Class: BSAI F23 Red
Student: B23F0063AI107

Business Problem: Customer Churn Prediction for Telecom Industry
Objective: Predict customer churn to help telecom companies reduce customer attrition and improve retention strategies
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

class CustomerChurnAnalysis:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("="*60)
        print("CUSTOMER CHURN PREDICTION ANALYSIS")
        print("="*60)
        
        # Load customer churn dataset
        try:
            # Try to load from URL first
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            self.df = pd.read_csv(url)
            print("✓ Dataset loaded successfully from IBM repository")
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
        
        # Churn distribution
        print("\n5. TARGET VARIABLE DISTRIBUTION:")
        print(self.df['Churn'].value_counts())
        print(f"Churn Rate: {self.df['Churn'].value_counts()['Yes'] / len(self.df) * 100:.1f}%")
        
        # Create visualizations
        self.create_exploratory_plots()
        
    def create_exploratory_plots(self):
        """Create comprehensive exploratory visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Churn Dataset - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # Churn distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Customer Churn Distribution')
        
        # Tenure distribution
        axes[0,1].hist(self.df['tenure'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Customer Tenure Distribution')
        axes[0,1].set_xlabel('Tenure (months)')
        axes[0,1].set_ylabel('Frequency')
        
        # Monthly charges vs churn
        sns.boxplot(data=self.df, x='Churn', y='MonthlyCharges', ax=axes[0,2])
        axes[0,2].set_title('Monthly Charges by Churn Status')
        axes[0,2].set_xlabel('Churn')
        axes[0,2].set_ylabel('Monthly Charges ($)')
        
        # Contract type vs churn
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'])
        contract_churn.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Churn by Contract Type')
        axes[1,0].set_xlabel('Contract Type')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend(title='Churn')
        
        # Internet service vs churn
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'])
        internet_churn.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Churn by Internet Service')
        axes[1,1].set_xlabel('Internet Service')
        axes[1,1].set_ylabel('Count')
        axes[1,1].legend(title='Churn')
        
        # Total charges vs churn
        sns.boxplot(data=self.df, x='Churn', y='TotalCharges', ax=axes[1,2])
        axes[1,2].set_title('Total Charges by Churn Status')
        axes[1,2].set_xlabel('Churn')
        axes[1,2].set_ylabel('Total Charges ($)')
        
        plt.tight_layout()
        plt.savefig('customer_churn_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Remove customer ID (not useful for prediction)
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
        
        # Handle missing values in TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(self.df['TotalCharges'].median())
        
        # Encode categorical variables
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.drop('Churn')  # Don't encode target yet
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        self.df['Churn'] = le_target.fit_transform(self.df['Churn'])
        self.label_encoders['Churn'] = le_target
        
        print("Categorical variables encoded:")
        for col in categorical_columns:
            print(f"  {col}: {len(self.label_encoders[col].classes_)} categories")
        
        # Feature selection
        feature_cols = [col for col in self.df.columns if col != 'Churn']
        X = self.df[feature_cols]
        y = self.df['Churn']
        
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
        
        # Check class distribution
        print(f"\nClass distribution in training set:")
        print(f"  No Churn: {(self.y_train == 0).sum()} ({(self.y_train == 0).mean()*100:.1f}%)")
        print(f"  Churn: {(self.y_train == 1).sum()} ({(self.y_train == 1).mean()*100:.1f}%)")
        
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
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
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
        best_precision = self.results[best_model_name]['precision']
        best_recall = self.results[best_model_name]['recall']
        
        print(f"\n1. MODEL PERFORMANCE SUMMARY:")
        print(f"   • Best Model: {best_model_name}")
        print(f"   • Accuracy: {best_accuracy:.1%}")
        print(f"   • Precision: {best_precision:.1%}")
        print(f"   • Recall: {best_recall:.1%}")
        print(f"   • This model can correctly predict customer churn {best_accuracy:.1%} of the time")
        
        print(f"\n2. BUSINESS APPLICATIONS:")
        print(f"   • Churn Prevention: Identify at-risk customers before they leave")
        print(f"   • Retention Campaigns: Target high-risk customers with retention offers")
        print(f"   • Customer Segmentation: Segment customers based on churn risk")
        print(f"   • Resource Allocation: Focus retention efforts on high-value at-risk customers")
        
        print(f"\n3. KEY FINDINGS:")
        print(f"   • Model achieves {best_accuracy:.1%} accuracy in churn prediction")
        print(f"   • This enables proactive customer retention strategies")
        print(f"   • Churn prediction can reduce customer acquisition costs")
        print(f"   • Enables data-driven customer relationship management")
        
        print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
        print(f"   • Deploy {best_model_name} model in customer management system")
        print(f"   • Integrate with CRM for real-time churn alerts")
        print(f"   • Train customer service teams on churn risk indicators")
        print(f"   • Monitor model performance and retrain quarterly")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Customer Churn Prediction Analysis...")
        
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
        print("• customer_churn_analysis.png - Exploratory analysis plots")
        print("• feature_importance_*.png - Feature importance plots")
        print("• This script contains all analysis code")

if __name__ == "__main__":
    # Run the complete analysis
    analysis = CustomerChurnAnalysis()
    analysis.run_complete_analysis()
