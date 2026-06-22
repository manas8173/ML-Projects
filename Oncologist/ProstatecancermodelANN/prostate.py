import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv(r'C:\Users\surya\.vscode\all in one\Prostate_Cancer.csv')
df = df.drop('id', axis=1)
df['diagnosis_result'] = df['diagnosis_result'].map({'M': 0, 'B': 1})

# Normalize numeric columns
cols_to_norm = ['perimeter', 'area', 'radius', 'texture']
scaler = MinMaxScaler()
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

# Features and target
X = df.drop('diagnosis_result', axis=1)
y = df['diagnosis_result']

# -----------------------------
# Stratified Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(class_weight='balanced', C=10, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Save the trained model
# -----------------------------
with open("prostate_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as prostate_cancer_model.pkl")

# -----------------------------
# Load the model back
# -----------------------------
with open("prostate_cancer_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
print("Model loaded from pickle file.")

# -----------------------------
# Predictions & Evaluation
# -----------------------------
y_pred = loaded_model.predict(X_test)
y_prob = loaded_model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report (Loaded Model):")
print(classification_report(y_test, y_pred))

# ROC curve & AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Prostate Cancer Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"AUC Score (Loaded Model): {roc_auc:.3f}")
