import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")


# --- Encode local image to base64 ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

base64_img = get_base64_image("image.jpg")

# --- CSS with gradient overlay ---
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                      url("data:image/jpg;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- Title ---
st.markdown("<h1 style='color: white; text-align: center;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.write("")

# --- Load CSV Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("movies_50_dataset.csv")
    df = df.dropna(subset=["genres", "description"])
    df["content"] = df["genres"].astype(str) + " " + df["description"].astype(str)
    return df

df = load_data()

# --- TF-IDF and Cosine Similarity ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Recommendation Function ---
def recommend_movie(title):
    if title not in df["title"].values:
        return []
    index = df[df["title"] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_5 = similarity_scores[1:6]
    return [df.iloc[i[0]]["title"] for i in top_5]

# --- Streamlit UI ---
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.subheader("🎯 Select a movie you like:")
movie_list = df["title"].sort_values().tolist()
selected_movie = st.selectbox("", movie_list)

if st.button("🔍 Recommend"):
    recommendations = recommend_movie(selected_movie)
    if recommendations:
        st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
        for movie in recommendations:
            st.markdown(f"- 🎬 {movie}")
    else:
        st.error("Movie not found or not enough data.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Show/Hide Genre Distribution Chart ---
st.markdown("---")
show_chart = st.checkbox("📊 Show Genre Distribution Chart")

if show_chart:
    st.subheader("📊 Genre Distribution (based on listed genres)")
    genre_counts = df["genres"].str.split('|').explode().value_counts()

    fig, ax = plt.subplots()
    genre_counts.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Movie Genre Distribution")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    st.pyplot(fig)
