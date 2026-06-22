# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from tqdm import tqdm
import joblib
import time

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv(r'Heart_Disease_Prediction.csv')

# Encode target column ('Absence' → 0, 'Presence' → 1)
df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# Normalize selected numeric columns
scaler = MinMaxScaler()
df[['Age', 'BP', 'Cholesterol', 'Max HR']] = scaler.fit_transform(df[['Age', 'BP', 'Cholesterol', 'Max HR']])

# Split into features and target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Simulated Training Progress Bar
# -----------------------------
print("\n🔄 Training Logistic Regression Model...")
for _ in tqdm(range(100), desc="Training Progress", ncols=80):
    time.sleep(0.02)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("\n---- Test Classification Report ----")
print(classification_report(y_test, y_pred_test))

print("---- Train Classification Report ----")
print(classification_report(y_train, y_pred_train))

# -----------------------------
# ROC Curve + AUC
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Heart Disease Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"\n✅ AUC Score: {roc_auc:.3f}")

# -----------------------------
# Save model and scaler
# -----------------------------
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 Model and scaler saved successfully!")
