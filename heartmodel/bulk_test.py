import numpy as np
import pandas as pd

np.random.seed(42)  # reproducible

columns = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# ----- 100 Normal Cases -----
normal = pd.DataFrame({
    'Age': np.random.randint(30,56,100),
    'Sex': np.random.randint(0,2,100),
    'Chest pain type': np.random.randint(1,4,100),
    'BP': np.random.randint(110,131,100),
    'Cholesterol': np.random.randint(170,241,100),
    'FBS over 120': 0,
    'EKG results': np.random.randint(0,2,100),
    'Max HR': np.random.randint(150,181,100),
    'Exercise angina': 0,
    'ST depression': np.random.uniform(0,0.5,100).round(2),
    'Slope of ST': np.random.randint(1,3,100),
    'Number of vessels fluro': np.random.randint(0,2,100),
    'Thallium': np.random.randint(3,8,100)
})
normal['Expected Output'] = 'No'

# ----- 100 Detection Cases -----
detection = pd.DataFrame({
    'Age': np.random.randint(55,76,100),
    'Sex': np.random.randint(0,2,100),
    'Chest pain type': np.random.randint(3,5,100),
    'BP': np.random.randint(140,181,100),
    'Cholesterol': np.random.randint(250,361,100),
    'FBS over 120': 1,
    'EKG results': 2,
    'Max HR': np.random.randint(100,131,100),
    'Exercise angina': 1,
    'ST depression': np.random.uniform(1.5,3.5,100).round(2),
    'Slope of ST': np.random.randint(2,4,100),
    'Number of vessels fluro': np.random.randint(1,4,100),
    'Thallium': np.random.randint(3,8,100)
})
detection['Expected Output'] = 'Yes'

# Combine into a single DataFrame
bulk_df = pd.concat([normal,detection], ignore_index=True)

# Shuffle the rows
bulk_df = bulk_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV for testing
bulk_df.to_csv('bulk_200_test_cases.csv', index=False)

print("✅ 200 test cases generated (100 Normal + 100 Detection)")
