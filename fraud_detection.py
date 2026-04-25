# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)

from imblearn.over_sampling import SMOTE

# LOAD DATASET
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# CHECK MISSING VALUES
print(df.isnull().sum())

# VISUALIZE ORIGINAL CLASS DISTRIBUTION
plt.title("Original Class Distribution (Highly Imbalanced)")
sns.countplot(x='Class', data=df)
plt.show()

# DATA PREPROCESSING
X = df.drop("Class", axis=1)
y = df["Class"]

# SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# HANDLE IMBALANCE USING SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

print("Before SMOTE:\n", y.value_counts())
print("After SMOTE:\n", pd.Series(y_res).value_counts())

# VISUALIZE BALANCED DATA
sns.countplot(x=y_res)
plt.title("Balanced Class Distribution (After SMOTE)")
plt.show()

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# EVALUATION FUNCTION
def evaluate_model(y_true, y_pred, model_name, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred))

    # CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC CURVE
    roc_auc = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend()
        plt.show()

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1, "ROC-AUC": roc_auc}


# TRAIN & EVALUATE
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = evaluate_model(y_test, y_pred, name, y_proba)

# RESULTS COMPARISON
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

results_df[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# PREDICT ON NEW DATA (example)
new_data = X_test[:5]
predictions = models["Random Forest"].predict(new_data)

print("Predicted classes for new transactions:", predictions)