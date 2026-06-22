import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    # ✅ Scale using the same scaler used in training
    final_scaled = scaler.transform(final_features)

    prediction = model.predict(final_scaled)
    output = 'Presence' if prediction[0] == 1 else 'Absence'

    return render_template('index.html', prediction_text=f'Heart Disease: {output}')

if __name__ == '__main__':
    app.run(debug=True)
