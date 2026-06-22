import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, roc_auc_score
)
import pickle

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv(r'C:\Users\surya\.vscode\all in one\breast.csv')

# Encode target column ('M' → 0, 'B' → 1)
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
df = df.drop('id', axis=1)

# Normalize selected numeric columns
norm = MinMaxScaler()
cols_to_norm = ['perimeter_mean', 'area_mean', 'area_se', 'perimeter_worst', 'area_worst']
df[cols_to_norm] = norm.fit_transform(df[cols_to_norm])

# Split into features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Save the trained model
# -----------------------------
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as breast_cancer_model.pkl")

# -----------------------------
# Load the model back
# -----------------------------
with open("breast_cancer_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
print("Model loaded from pickle file.")

# -----------------------------
# Predictions & Evaluation
# -----------------------------
# Predict probabilities for positive class
y_prob = loaded_model.predict_proba(X_test)[:, 1]
y_pred = loaded_model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC curve & AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Breast Cancer')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"AUC Score: {roc_auc:.3f}")
