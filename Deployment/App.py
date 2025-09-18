import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)
# --------------------------
# TMDb & OMDB API Config
# --------------------------
TMDB_API_KEY = "c67407b2ec323db843bf5d5cde43ccec"
OMDB_API_KEY = "d490fe65"
TMDB_BASE = "https://api.themoviedb.org/3/find/{}?api_key={}&external_source=imdb_id"
OMDB_BASE = "http://www.omdbapi.com/?i={}&apikey={}"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
FALLBACK_URL = "https://via.placeholder.com/500x750?text=No+Poster"

# --------------------------
# Load Pickle Data
# --------------------------
@st.cache_resource
def load_model():
    with open("Deployment/model.joblib", "rb") as f:
        data = joblib.load(f)
    return data

data = load_model()
movies = data["movies"]
X_full = data["X_full"]
indices = data["indices"]

# --------------------------
# Poster Fetch Function
# --------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(imdb_id):
    if pd.isna(imdb_id) or imdb_id == "":
        return FALLBACK_URL
    try:
        response = requests.get(TMDB_BASE.format(imdb_id, TMDB_API_KEY)).json()
        results = response.get("movie_results")
        if results and results[0].get("poster_path"):
            return IMG_BASE + results[0]["poster_path"]
    except:
        pass
    try:
        response = requests.get(OMDB_BASE.format(imdb_id, OMDB_API_KEY)).json()
        poster = response.get("Poster")
        if poster and poster != "N/A":
            return poster
    except:
        pass
    return FALLBACK_URL

# --------------------------
# Recommendation Function
# --------------------------
def recommend(title, top_n=20):
    title = title.strip()
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    root = str(movies.loc[idx, "root_title"])
    same_series = movies[movies["root_title"] == root].copy()
    same_series = same_series[same_series["title"] != title]

    target_vec = X_full[idx]
    sims = cosine_similarity(target_vec, X_full).flatten()
    sims[idx] = -1  

    top_indices = sims.argsort()[::-1]
    top_indices = top_indices[top_indices < len(movies)]

    recs = movies.iloc[top_indices][['title','release_year','vote_average','imdb_id']].copy()
    recs["similarity"] = sims[top_indices]

    results = pd.concat([
        same_series[['title','release_year','vote_average','imdb_id']].assign(similarity=1.0),
        recs
    ])
    results = results.drop_duplicates(subset="title").head(top_n).reset_index(drop=True)
    return results

# --------------------------
# Streamlit App
# --------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Find similar movies and explore top movies by genre.")

page = st.sidebar.radio("📂 Choose Page:", ["Find Similar Movies", "Explore by Genre"])

# --------------------------
# Display Function with Progress
# --------------------------
def display_movies_grid(df, n_cols=5):
    rows = df.to_dict(orient="records")
    progress = st.progress(0)
    total = len(rows)
    for i in range(0, total, n_cols):
        cols = st.columns(n_cols)
        batch = rows[i:i+n_cols]
        for j, movie in enumerate(batch):
            with cols[j]:
                st.image(fetch_poster(movie['imdb_id']), width=150)
                st.markdown(f"**{movie['title']} ({movie['release_year']})**")
                if 'similarity' in movie:
                    st.write(f"⭐ {movie['vote_average']} | 🔗 {movie['similarity']:.2f}")
                else:
                    st.write(f"⭐ {movie['vote_average']} | 🔥 {movie.get('vote_count', 0)}")
        progress.progress(min((i+n_cols)/total,1.0))
        time.sleep(0.1)  # صغير لتحديث الشاشة بسلاسة
    progress.empty()

# --------------------------
# Page 1: Find Similar Movies
# --------------------------
if page == "Find Similar Movies":
    st.header("🔍 Find Similar Movies")
    movie_choice = st.selectbox("🎬 Select a movie:", movies['title'].values)
    if movie_choice:
        results = recommend(movie_choice, top_n=20)
        if results.empty:
            st.warning("⚠️ Title not found or no recommendations.")
        else:
            st.subheader(f"Top 20 Movies Similar to: {movie_choice}")
            display_movies_grid(results, n_cols=5)

# --------------------------
# Page 2: Explore by Genre
# --------------------------
elif page == "Explore by Genre":
    st.header("📊 Explore Movies by Genre")
    genre_columns = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
        'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 
        'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 
        'Thriller', 'War', 'Western'
    ]
    selected_genre = st.selectbox("🎭 Select Genre:", genre_columns)
    if selected_genre:
        genre_movies = movies[movies[selected_genre] == 1]
        top20 = genre_movies[genre_movies['vote_count'] > 50] \
                    .sort_values(by="vote_average", ascending=False) \
                    .head(20)
        st.subheader(f"🎭 Top 20 {selected_genre} Movies")
        display_movies_grid(top20, n_cols=5)








