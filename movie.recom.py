# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import requests
from sklearn.neighbors import NearestNeighbors
from typing import List

# -----------------------------
# App config / defaults
# -----------------------------
APP_TITLE = "🎬 CineScope — AI Movie Recommender"
OWNER_NAME = "Chandan Kumar Sah"
CONTACT_EMAIL = "irisblack0503@gmail.com"
GITHUB_URL = "https://github.com/ChankumarSah"
LINKEDIN_URL = "https://linkedin.com/in/chandan-kumar-sah-752803387"
DEFAULT_OMDB_KEY = "8fecb11c"
PLACEHOLDER_IMG = "https://via.placeholder.com/220x330?text=No+Poster"

DF_FILE = "df.pkl"
VECTORS_FILE = "vectors.pkl"
MODEL_FILE = "model.pkl"
APP_VERSION = "v1.0"
DATASET_NAME = "movies_content.csv"

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

# -----------------------------
# CSS Theme (NO f-string here!)
# -----------------------------
css = """
<style>
.stApp {
  background:
    radial-gradient(1200px 600px at 10% 10%, rgba(106,0,255,0.06), transparent 10%),
    radial-gradient(1000px 500px at 90% 90%, rgba(255,68,102,0.04), transparent 10%),
    linear-gradient(180deg, #020617 0%, #071A52 50%, #0f1030 100%);
  color: #e6eef8;
  font-family: "Segoe UI", Roboto, Arial, sans-serif;
}

/* Header section */
.app-header {
  padding: 28px 22px;
  border-radius: 14px;
  color: white;
  text-align: center;
  box-shadow: 0 14px 40px rgba(2,6,23,0.6);
}
.app-title { font-size: 36px; font-weight:900; margin-bottom:6px; }
.app-sub { font-size:15px; opacity:0.95; font-weight:600; }

/* Card */
.card {
  padding: 14px;
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  box-shadow: 0 8px 24px rgba(2,6,23,0.5);
  border: 1px solid rgba(255,255,255,0.03);
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, #ff8c00, #ff4466);
  color: #fff;
  padding: 10px 18px;
  border-radius: 12px;
  font-weight: 700;
  border: none;
  box-shadow: 0 8px 24px rgba(106,0,255,0.12);
}
.stButton>button:hover { transform: translateY(-2px); }

/* Rounded images */
img { border-radius: 8px; }

/* Inputs */
.stSelectbox>div, .stTextInput>div {
  border-radius: 10px;
}
</style>
"""
components.html(css, height=0)

# -----------------------------
# HEADER  (This uses fSTRING – allowed!)
# -----------------------------
# header snippet — paste into your app.py
APP_TITLE = "🎬 CineScope — Curated Movie Picks"
OWNER_NAME = "Chandan Kumar Sah"
LINKEDIN_URL = "https://linkedin.com/in/chandan-kumar-sah-752803387"
GITHUB_URL = "https://github.com/ChankumarSah"

header_html = f"""
<style>
  .app-header {{
    background: linear-gradient(90deg, #071A52 0%, #6A00FF 50%, #FFD27A 100%);
    padding: 26px 22px;
    border-radius: 14px;
    color: #fff;
    text-align: center;
    box-shadow: 0 12px 36px rgba(2,6,23,0.55);
    font-family: "Segoe UI", Roboto, Arial, serif;
    overflow: hidden;
  }}
  .app-title {{ font-size: 36px; font-weight: 900; margin-bottom: 6px; letter-spacing:0.4px; color: #fff; }}
  .app-sub {{ font-size: 15px; margin-top: 6px; font-weight: 600; color: rgba(255,255,255,0.92); }}
  .gold-accent {{ width:86px; height:6px; margin:10px auto; border-radius:6px; background: linear-gradient(90deg,#FFD27A,#C49A6C); box-shadow: 0 6px 18px rgba(255,210,122,0.08); }}
  .author-badge {{
    margin-top: 12px;
    padding: 8px 16px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    font-size:14px;
    font-weight:700;
    display:inline-block;
    color: #fff;
  }}
  .pill-links {{ margin-top:12px; display:flex; justify-content:center; gap:10px; }}
  .pill-links a {{
    background: rgba(255,255,255,0.08);
    padding:8px 14px;
    border-radius:999px;
    color:white !important;
    text-decoration:none;
    font-weight:600;
    box-shadow: 0 6px 18px rgba(106,0,255,0.06);
  }}
  .pill-links a:hover {{ background: rgba(255,255,255,0.18); transform:translateY(-2px); }}
  @media(max-width:900px) {{
    .app-title {{ font-size: 26px; }}
    .app-sub {{ font-size: 13px; }}
  }}
</style>

<div class="app-header">
  <div class="app-title">{APP_TITLE}</div>
  <div class="gold-accent" aria-hidden="true"></div>
  <div class="app-sub">Discover movies like the ones you love — curated picks with pretty posters and instant results.</div>
  <div style="height:12px;"></div>
  <div class="author-badge">👨‍💻 {OWNER_NAME}</div>
  <div class="pill-links">
    <a href="{LINKEDIN_URL}" target="_blank" rel="noopener">🔗 LinkedIn</a>
    <a href="{GITHUB_URL}" target="_blank" rel="noopener">💻 GitHub</a>
  </div>
</div>
"""

components.html(header_html, height=240, scrolling=False)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 🎬 CineScope — Summary")
    st.write("AI-powered content-based movie recommendation system.")
    st.markdown("---")

    st.markdown("## ⚙️ Required Files")
    df_load_path = st.text_input("df.pkl path", DF_FILE)
    vectors_load_path = st.text_input("vectors.pkl path", VECTORS_FILE)
    model_load_path = st.text_input("model.pkl path", MODEL_FILE)
    api_key_input = st.text_input("OMDb API key", DEFAULT_OMDB_KEY)
    st.caption("Files must be generated from the same dataset.")
    st.markdown("---")

    st.markdown("## 🧠 How it works")
    st.write("- Combine description + genre + director + language")
    st.write("- Convert into text embeddings")
    st.write("- Calculate cosine similarity")
    st.write("- Return nearest movies")
    st.markdown("---")

    st.markdown("## 🧭 Tips")
    st.write("Choose 3–8 recommendations for best results.")
    st.write("Check OMDb API key if posters fail.")
    st.markdown("---")

    st.markdown("## Developer")
    st.write(f"**{OWNER_NAME}**")
    st.write(f"📧 {CONTACT_EMAIL}")
    st.markdown(f"[GitHub]({GITHUB_URL}) • [LinkedIn]({LINKEDIN_URL})")
    st.markdown("---")

    movie_count_placeholder = st.empty()

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_dataframe(path): return joblib.load(path)

@st.cache_data
def load_vectors(path): return joblib.load(path)

@st.cache_resource
def load_model(path): return joblib.load(path)

@st.cache_data
def fetch_movie_poster(imdb_id, api_key):
    try:
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        r = requests.get(url).json()
        if "Poster" in r and r["Poster"] != "N/A":
            return r["Poster"]
    except:
        pass
    return PLACEHOLDER_IMG

# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend_movies(movie_name, df, vectors, model, k=5):
    if movie_name not in df["name"].values:
        return []
    idx = int(df[df["name"] == movie_name].index[0])
    distances, indexes = model.kneighbors([vectors[idx]], n_neighbors=k+1)
    recs = []

    for dist, neighbor_idx in zip(distances[0][1:], indexes[0][1:]):
        recs.append({
            "movie_id": df.loc[neighbor_idx, "movie_id"],
            "name": df.loc[neighbor_idx, "name"],
            "distance": float(dist)
        })
    return recs

# -----------------------------
# Load resources
# -----------------------------
try:
    df = load_dataframe(df_load_path)
    vectors = load_vectors(vectors_load_path)
    model = load_model(model_load_path)
    movie_count_placeholder.markdown(f"**Movies Loaded:** {len(df)}")
except Exception as e:
    st.error("Could not load required files. Check paths.")
    st.stop()

# -----------------------------
# Main UI
# -----------------------------
st.markdown("### 🔎 Search for a movie")
movie_list = sorted(df["name"].unique())
movie_input = st.selectbox("Choose a movie", ["-- select a movie --"] + movie_list)

st.markdown("### 🎯 Number of recommendations")
k = st.selectbox("Select number", [1,2,3,4,5,6,8], index=4)

if st.button("Recommend Movies"):
    if movie_input == "-- select a movie --":
        st.warning("Please select a movie.")
    else:
        sel_idx = int(df[df["name"] == movie_input].index[0])
        sel_id = df.loc[sel_idx, "movie_id"]

        st.markdown("### 🎥 Selected Movie")
        st.image(fetch_movie_poster(sel_id, api_key_input), width=200, caption=movie_input)

        st.markdown("### ⭐ Recommendations")
        recs = recommend_movies(movie_input, df, vectors, model, k=k)

        cols = st.columns(4)
        for i, rec in enumerate(recs):
            with cols[i % 4]:
                st.image(fetch_movie_poster(rec["movie_id"], api_key_input), width=180)
                st.write(f"**{rec['name']}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    f"""
    <div style='text-align:center; color:#9aa8c3; padding:20px 0; font-size:13px;'>
        Built with ❤️ using Streamlit · Developed by <b>{OWNER_NAME}</b><br>
        <a href="{LINKEDIN_URL}" style="color:#9aa8c3;">LinkedIn</a> |
        <a href="{GITHUB_URL}" style="color:#9aa8c3;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
