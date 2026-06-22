import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import pickle

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv(r'C:\Users\surya\.vscode\all in one\survey lung cancer2.csv')

# Encode target variable ('YES' -> 1, 'NO' -> 0)
df['Lung Cancer'] = df['Lung Cancer'].map({'YES': 1, 'NO': 0})

# Encode categorical columns
cols_to_encode = ['Gender', 'Smoking', 'Yellow Fingers', 'Anxiety', 'Peer Presure',
                  'Chronic Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol Consuming',
                  'Coughing', 'Shortness of breath', 'Swallowing Difficulty', 'Chest Pain']
le = LabelEncoder()
df[cols_to_encode] = df[cols_to_encode].apply(le.fit_transform)

# Normalize the 'Age' column
scaler = MinMaxScaler()
df[['Age']] = scaler.fit_transform(df[['Age']])

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop('Lung Cancer', axis=1)
y = df['Lung Cancer']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Save the trained model
# -----------------------------
with open("lung_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as lung_cancer_model.pkl")

# -----------------------------
# Load the model back
# -----------------------------
with open("lung_cancer_model.pkl", "rb") as f:
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
plt.title('ROC Curve - Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"AUC Score (Loaded Model): {roc_auc:.3f}")
