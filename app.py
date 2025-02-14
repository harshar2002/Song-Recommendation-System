import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Set page title and layout
st.set_page_config(page_title="🎵 Music Recommender", page_icon="🎧", layout="wide")

# Load dataset & scaled features (Use caching to prevent reloading)
@st.cache_data
def load_data():
    df_cleaned = pd.read_csv("df_cleaned.csv")
    df_scaled = np.load("df_scaled.npy")
    return df_cleaned, df_scaled

df_cleaned, df_scaled = load_data()

# Drop 'track_id' column if it exists
if 'track_id' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['track_id'])

df_cleaned["song_display"] = df_cleaned["track_name"] + " - " + df_cleaned["artists"]

# Last.fm API Key
LASTFM_API_KEY = "fa10f0c463273e74e58eebf856d5df9d"

# Define the local fallback image
LOCAL_IMAGE_PATH = "images.jpg"

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; font-family: 'Arial', sans-serif; }
    .title-container { text-align: center; margin-top: -40px; margin-bottom: 10px; }
    .title-container h1 { font-size: 42px; font-weight: bold; color: #1DB954; }
    .title-container p { font-size: 18px; color: #BBBBBB; margin-bottom: 0px; }
    .stButton>button { background-color: #1DB954 !important; color: white !important; font-size: 18px !important; border-radius: 10px; padding: 10px 20px; }
    .recommend-title { font-size: 24px; font-weight: bold; color: #1DB954; margin-bottom: 15px; text-align: center; }
    .song-card { background-color: #222; padding: 15px; border-radius: 12px; border: 1px solid #444; text-align: center; width: 240px; height: 260px; transition: transform 0.3s ease, box-shadow 0.3s ease; }
    .song-card:hover { transform: scale(1.05); box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.2); }
    .footer { text-align: center; padding: 15px; margin-top: 40px; background: linear-gradient(to right, #1DB954, #191414); color: white; font-size: 16px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3); }
    .footer a { color: #1DB954; font-weight: bold; text-decoration: none; transition: color 0.3s ease; }
    .footer a:hover { color: #ffffff; text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

# Compute cosine similarity
@st.cache_data
def compute_cosine_similarity(index, df_scaled):
    if index >= len(df_scaled):
        return np.zeros(len(df_scaled))
    return cosine_similarity(df_scaled[index].reshape(1, -1), df_scaled).flatten()

def content_based_recommend(song_name, top_n=10):
    if song_name not in df_cleaned['track_name'].values:
        return None
    index = df_cleaned[df_cleaned["track_name"] == song_name].index[0]
    similarity_scores = compute_cosine_similarity(index, df_scaled)
    similar_song_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    return df_cleaned.iloc[similar_song_indices][['track_name', 'artists', 'album_name', 'track_genre']]

@st.cache_data
def get_album_art(song_name, artist_name):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={artist_name}&track={song_name}&format=json"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        data = response.json()
        if "track" in data and "album" in data["track"] and "image" in data["track"]["album"]:
            album_image = data["track"]["album"]["image"][-1]["#text"]
            if album_image:  
                return album_image
    except (requests.exceptions.RequestException, KeyError):
        pass
    return LOCAL_IMAGE_PATH

# UI Components
st.markdown('<div class="title-container"><h1>🎵 Music Recommender System</h1><p>Find songs similar to your favorites</p></div>', unsafe_allow_html=True)

song_options = ["Select a Song"] + df_cleaned["song_display"].unique().tolist()
selected_song = st.selectbox("", options=song_options, index=0, format_func=lambda x: "🔍 " + x if x != "Select a Song" else "🎶 Select a Song", key="song_selection")

if selected_song != "Select a Song":
    song_name = df_cleaned[df_cleaned["song_display"] == selected_song]["track_name"].values[0]
    artist_name = df_cleaned[df_cleaned["song_display"] == selected_song]["artists"].values[0]

    if st.button("🔍 Get Recommendations"):
        with st.spinner("🔄 Fetching recommendations..."):
            recommendations = content_based_recommend(song_name)
            if recommendations is None or recommendations.empty:
                st.error(f"'{song_name}' not found in the dataset. Please enter a valid song.")
            else:
                st.markdown('<div class="recommend-title">🎼 Recommended Songs</div>', unsafe_allow_html=True)
                cols = st.columns(5)
                for i in range(min(10, len(recommendations))):
                    row = recommendations.iloc[i]
                    album_art_url = get_album_art(row['track_name'], row['artists'])
                    with cols[i % 5]:
                        st.image(album_art_url, width=150)
                        st.markdown(f"""<div class="song-card"><h3>{row['track_name']}</h3><h4>{row['artists']}</h4><p>{row['album_name']}</p><p><i>{row['track_genre']}</i></p></div>""", unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        🎵 Made with ❤️ by <a href="https://github.com/harshar2002" target="_blank">Harsha</a> | 
        📩 Contact: <a href="mailto:harshagowda497@gmail.com">harshagowda497@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
