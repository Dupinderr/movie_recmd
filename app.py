import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Set Streamlit Page ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# --- Background Image Styling ---
st.markdown("""
    <style>
    .stApp {
        background-image: url('image.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3, .stMarkdown, .stText, .stSelectbox, .stButton {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🎥 Movie Recommendation System")

# --- Movie Dataset ---
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

df = pd.DataFrame(data)

# --- Fix: Convert columns to string before concatenation ---
df["content"] = df["genres"].astype(str).fillna('') + " " + df["description"].astype(str).fillna('')

# --- TF-IDF and Cosine Similarity ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Recommend Function ---
def recommend_movie(title):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]
    return df.iloc[top_indices]['title'].tolist()

# --- UI Interaction ---
st.markdown("### 💡 Select a movie to get recommendations")
movie = st.selectbox("🎬 Choose a movie", df['title'])

if st.button("Get Recommendations"):
    recs = recommend_movie(movie)
    if recs:
        st.subheader(f"🎯 Because you liked '{movie}', you may also like:")
        for i, r in enumerate(recs, 1):
            st.markdown(f"**{i}.** {r}")
    else:
        st.warning("No recommendations found.")
