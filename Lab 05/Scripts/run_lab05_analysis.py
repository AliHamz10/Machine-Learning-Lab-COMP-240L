#!/usr/bin/env python3
"""
Lab 05: Logistic Regression Analysis Runner
Automatically executes the complete logistic regression analysis
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def run_notebook():
    """Execute the Jupyter notebook"""
    print("Running Lab 05 Logistic Regression Analysis...")
    try:
        # Convert notebook to Python and execute
        subprocess.check_call([
            "jupyter", "nbconvert", 
            "--to", "python", 
            "--execute", 
            "../Analysis/Lab_05_Logistic_Regression_Analysis.ipynb",
            "--ExecutePreprocessor.timeout=300"
        ])
        print("✓ Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running analysis: {e}")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("LAB 05: LOGISTIC REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("../Analysis/Lab_05_Logistic_Regression_Analysis.ipynb"):
        print("✗ Error: Lab_05_Logistic_Regression_Analysis.ipynb not found!")
        print("Please run this script from the Scripts directory.")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Run the analysis
    if not run_notebook():
        return False
    
    print("\n" + "=" * 60)
    print("LAB 05 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- Lab_05_Logistic_Regression_Analysis.py (executed notebook)")
    print("- Various visualization plots")
    print("\nTo view the results, open Analysis/Lab_05_Logistic_Regression_Analysis.ipynb in Jupyter")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
