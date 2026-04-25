# Credit-card-fraud-detection

## 📌 Overview
This project detects fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset.

## ⚙️ Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## 📊 Dataset
- Kaggle Credit Card Fraud Detection dataset
- ~284,000 transactions with severe class imbalance

## 🚀 Features
- Data preprocessing and feature scaling
- Handling imbalance using SMOTE
- Model comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluation using:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC Curve

## 🏆 Results
- Random Forest performed best
- Accuracy: ~99.9%
- Recall: ~0.92 (important for fraud detection)

## ▶️ How to Run
```bash
pip install -r requirements.txt
python fraud_detection.py
