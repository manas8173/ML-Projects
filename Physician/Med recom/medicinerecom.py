import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# -------------------------------
# Step 1: Load Data
# -------------------------------
company_df = pd.read_excel(r"Company_Name.xlsx")
medicine_df = pd.read_excel(r"Medicine_description.xlsx")
rating_df = pd.read_excel(r"Ratings.xlsx")

# -------------------------------
# Step 2: Merge Data
# -------------------------------
merged = medicine_df.copy()
merged['Short-form'] = merged['Drug_Name'].apply(lambda x: x.split()[0][:3].upper() if isinstance(x, str) else "")
merged = merged.merge(rating_df, on='Short-form', how='left')

# Fill missing ratings with average
merged['Rating'] = merged['Rating'].fillna(merged['Rating'].mean())

# -------------------------------
# Step 3: Create TF-IDF Matrix for Description + Reason
# -------------------------------
merged['text'] = merged['Reason'].astype(str) + " " + merged['Description'].astype(str)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(merged['text'])

# -------------------------------
# Step 4: Medicine Recommendation Function
# -------------------------------
def recommend_medicine(symptom, top_n=5):
    """Return top_n medicines for a given symptom."""
    # Convert user input to vector
    symptom_vec = vectorizer.transform([symptom])
    sim_scores = cosine_similarity(symptom_vec, tfidf_matrix).flatten()

    merged['similarity'] = sim_scores

    # Combine similarity and rating for ranking
    merged['final_score'] = merged['similarity'] * 0.7 + (merged['Rating'] / 5) * 0.3

    top_results = merged.sort_values(by='final_score', ascending=False).head(top_n)
    return top_results[['Drug_Name', 'Reason', 'Description', 'Rating', 'final_score']]


# Save trained objects
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(merged, "medicine_data.pkl")

print("✅ Model and data saved successfully!")

# -------------------------------
# Step 5: Test Example
# -------------------------------
if __name__ == "__main__":
    symptom_input = input("Enter symptom or reason (e.g., leg pain, acne, fever): ")
    recommendations = recommend_medicine(symptom_input)
    print("\nTop Recommended Medicines:\n")
    print(recommendations.to_string(index=False))