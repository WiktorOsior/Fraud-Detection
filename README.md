# Financial Fraud Detection - Machine Learning Pipeline

This repository contains a comprehensive pipeline for detecting fraudulent transactions using a multi-layered data science approach. The project evaluates multiple models, ranging from traditional decision trees to state-of-the-art time-series deep learning.

## 📌 Project Overview
Fraud detection is a classic imbalanced classification problem. This project utilizes a dataset containing customer profiles, transaction metadata, and merchant info to identify suspicious activities.

### Key Features of the Pipeline:
* **Data Integration:** Merging multiple CSV sources (Transactions, Account Activity, Customer Profiles, etc.).
* **Advanced EDA:** Statistical analysis of transaction densities, correlations (Fraud vs. Non-Fraud), and feature distributions.
* **Feature Engineering:** Creation of time-based features (Hour, Day, Weekends), Time-since-last-login, and Transaction Amount Ratios.
* **Class Imbalance Handling:** Implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set.
* **Modeling:**
    * **Baseline:** Decision Tree & Random Forest.
    * **Boosting:** XGBoost with Hyperparameter Tuning.
    * **Sequential/Deep Learning:** RNN and LSTM (Long Short-Term Memory) models using **Masking** for variable sequence lengths.
    * **State-of-the-Art:** **ROCKET** (Random Convolutional Kernel Transform) for time-series classification.

## Getting Started

### Prerequisites
* Python 3.8+
* Kaggle API Credentials (for dataset downloading)
