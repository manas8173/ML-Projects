import pickle
import pandas as pd

# Load model & scaler
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# Define feature columns
columns = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# Load test cases CSV
df = pd.read_csv('bulk_200_test_cases.csv')

# Extract features for prediction
X = df[columns]
X_scaled = scaler.transform(X)

# Predict
df['Prediction'] = model.predict(X_scaled)
df['Prediction_YN'] = df['Prediction'].map({1:'Yes', 0:'No'})

# Calculate accuracy
accuracy = (df['Expected Output'] == df['Prediction_YN']).mean() * 100
print(f"Model Accuracy on 200 cases: {accuracy:.2f}%")

# Optional: Save predictions
df.to_csv('bulk_200_test_cases_with_predictions.csv', index=False)
