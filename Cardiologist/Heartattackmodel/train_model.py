import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# 1️⃣ Load data
data = pd.read_csv('Heart_Disease_Prediction.csv')

# 2️⃣ Split features and target
X = data[[
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]]
y = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})  # convert to 1/0

# 3️⃣ Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# 5️⃣ Save model and scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Model and scaler saved successfully!")
