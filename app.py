import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

# --- Background Image CSS ---
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1601933470928-c1f4ebaa45a3?auto=format&fit=crop&w=1650&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='color: white; text-align: center;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.write("")

# --- Movie Dataset ---
data = {
    'title': [...],  # replace [...] with your 50 titles
    'genres': [...],  # your genres list
    'description': [...]  # your descriptions
}

df = pd.DataFrame(data)

# Combine genres and descriptions
df["content"] = df["genres"].fillna('') + " " + df["description"].fillna('')

# TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Recommendation Function ---
def recommend_movie(title):
    if title not in df['title'].values:
        return []

    index = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_5 = similarity_scores[1:6]
    return [df.iloc[i[0]]["title"] for i in top_5]

# --- Streamlit UI ---
st.subheader("🎯 Select a movie you like:")

movie_list = df["title"].sort_values().tolist()
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("🔍 Recommend"):
    recommendations = recommend_movie(selected_movie)
    if recommendations:
        st.success("Because you liked **{}**, you might also enjoy:".format(selected_movie))
        for movie in recommendations:
            st.markdown(f"- 🎬 {movie}")
    else:
        st.error("Movie not found or not enough data.")

# --- Genre Distribution Chart ---
st.markdown("---")
st.subheader("📊 Genre Distribution (based on listed genres)")
genre_counts = df["genres"].str.split('|').explode().value_counts()

fig, ax = plt.subplots()
genre_counts.plot(kind="bar", color="skyblue", ax=ax)
ax.set_title("Movie Genre Distribution")
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
st.pyplot(fig)
