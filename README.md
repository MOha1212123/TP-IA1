
# M115 Lab 1: Phishing URL Detection

## Overview
This lab focuses on detecting phishing URLs using Machine Learning techniques and Explainable AI (XAI).

Dataset: PhiUSIIL Phishing URL Dataset
- 235,795 URLs
- 56 features
- Target: IsPhishing (1 = Phishing, 0 = Benign)

## Part 1: Exploratory Data Analysis (EDA)
- Exercise 1: Data quality inspection and target alignment.
- Exercise 2: Class balance and feature distributions.
- Exercise 3: Feature selection and correlation analysis.

## Part 2: Data Preprocessing
- Exercise 4: Stratified Train/Test Split (75% / 25%).
- StandardScaler for feature normalization.
- Prevention of data leakage.

## Part 3: Logistic Regression
- Exercise 5: Comparison of five Logistic Regression models.
- Exercise 6: L1 Regularization and feature selection.
- Exercise 7: Model coefficients and Odds Ratios.
- Comparison of L1 and L2 regularization.

## Part 4: Explainable AI (SHAP)
- Exercise 8: Global SHAP explanations.
- Exercise 9: Local explanations using Waterfall plots.
- Exercise 10: Decision threshold analysis (0.20, 0.50, 0.85).

## SOC Engineering Synthesis
- Hyperparameter and sparsity trade-off.
- Global model interpretability.
- SOC alert triage.
- False Positive analysis.
- Recommended decision threshold policy.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP

## Dataset
PhiUSIIL Phishing URL Dataset.

## Objective
Build an interpretable Machine Learning model capable of detecting phishing URLs and supporting SOC security operations.
