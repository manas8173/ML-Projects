from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------
# Load Saved Model and Data
# ---------------------------
vectorizer = joblib.load("vectorizer.pkl")
merged = joblib.load("medicine_data.pkl")

# ---------------------------
# Medicine Recommendation Function
# ---------------------------
def recommend_medicine(symptom, top_n=5):
    symptom_vec = vectorizer.transform([symptom])
    sim_scores = cosine_similarity(symptom_vec, vectorizer.transform(merged['text'])).flatten()

    merged['similarity'] = sim_scores
    merged['final_score'] = merged['similarity'] * 0.7 + (merged['Rating'] / 5) * 0.3

    top_results = merged.sort_values(by='final_score', ascending=False).head(top_n)
    return top_results[['Drug_Name', 'Reason', 'Description', 'Rating', 'final_score']]

# ---------------------------
# API Route for AJAX call
# ---------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    symptom = data.get('symptom', '')
    if not symptom.strip():
        return jsonify({"error": "No symptom provided"}), 400

    results = recommend_medicine(symptom)
    return jsonify(results.to_dict(orient='records'))

# ---------------------------
# Home Route (Optional)
# ---------------------------
@app.route('/')
def home():
    return render_template("index.html")  # your HTML page

if __name__ == "__main__":
    app.run(debug=True)
