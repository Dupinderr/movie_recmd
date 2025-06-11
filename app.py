import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Set page config ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# --- Add background image with gradient ---
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                        url('image.jpg');
            background-size: cover;
            background-position: center;
            color: white;
        }}
        .css-1d391kg {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --- App Title ---
st.title("🎥 Movie Recommendation System")

# --- Sample Movie Dataset ---
data = {
    'title': [
        'Inception', 'Interstellar', 'The Matrix', 'The Dark Knight', 'Pulp Fiction',
        'Forrest Gump', 'Fight Club', 'The Shawshank Redemption', 'The Godfather', 'Parasite'
    ],
    'genres': [
        'Action|Sci-Fi', 'Adventure|Drama|Sci-Fi', 'Action|Sci-Fi',
        'Action|Crime|Drama', 'Crime|Drama',
        'Drama|Romance', 'Drama', 'Drama', 'Crime|Drama', 'Thriller|Drama'
    ],
    'description': [
        'A thief steals secrets through dream-sharing technology.',
        'Explorers travel through a wormhole in space.',
        'A hacker discovers the nature of his reality.',
        'Batman faces the Joker who causes chaos in Gotham.',
        'Mobsters, a boxer, and a gangster’s wife in crime tales.',
        'Life journey of a kind-hearted man with low IQ.',
        'An office worker joins an underground fight club.',
        'Two imprisoned men bond and find redemption.',
        'An aging crime boss transfers control to his son.',
        'A poor family infiltrates a rich household.'
    ]
}

# --- Convert to DataFrame ---
df = pd.DataFrame(data)

# --- Combine genres and description for content-based filtering ---
df["genres"] = df["genres"].astype(str)
df["description"] = df["description"].astype(str)
df["content"] = df["genres"].fillna('') + " " + df["description"].fillna('')

# --- Vectorization ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])

# --- Cosine similarity matrix ---
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Movie Recommendation Function ---
def recommend_movie(title):
    if title not in df['title'].values:
        return []
    index = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in similarity_scores[1:6]]
    return df.iloc[top_indices]["title"].tolist()

# --- Streamlit Interface ---
st.markdown("### 💡 Select a movie to get similar recommendations")

movie_list = df['title'].tolist()
selected_movie = st.selectbox("🎬 Choose a movie", movie_list)

if st.button("Get Recommendations"):
    recommendations = recommend_movie(selected_movie)
    if recommendations:
        st.subheader(f"🎯 Because you liked '{selected_movie}', you may also like:")
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** {movie}")
    else:
        st.warning("No recommendations found.")
