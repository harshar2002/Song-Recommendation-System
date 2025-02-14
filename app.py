import streamlit as st
import pandas as pd
import numpy as np
import requests
import concurrent.futures
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

# Custom CSS for Modern UI & Footer Animation
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; font-family: 'Arial', sans-serif; }

    /* Centered Main Title */
    .title-container { text-align: center; margin-top: -40px; margin-bottom: 10px; }
    .title-container h1 { font-size: 42px; font-weight: bold; color: #1DB954; }
    .title-container p { font-size: 18px; color: #BBBBBB; margin-bottom: 0px; }

    .stButton>button { background-color: #1DB954 !important; color: white !important; font-size: 18px !important; border-radius: 10px; padding: 10px 20px; }

    /* Centered Recommended Songs */
    .recommend-title { 
        font-size: 24px; 
        font-weight: bold; 
        color: #1DB954; 
        margin-bottom: 15px; 
        text-align: center; 
    }

    /* Footer Section */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #181818;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #444;
        animation: fadeIn 2s ease-in-out;
    }
    
    /* Footer Links */
    .footer a {
        color: #1DB954;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    /* Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Compute cosine similarity (Optimize with caching)
@st.cache_data
def compute_cosine_similarity(index, df_scaled):
    if index >= len(df_scaled):
        return np.zeros(len(df_scaled))
    return cosine_similarity(df_scaled[index].reshape(1, -1), df_scaled).flatten()

# Recommendation Function (Returns 10 Recommendations)
def content_based_recommend(song_name, top_n=10):
    if song_name not in df_cleaned['track_name'].values:
        return None

    index = df_cleaned[df_cleaned["track_name"] == song_name].index[0]
    similarity_scores = compute_cosine_similarity(index, df_scaled)
    similar_song_indices = similarity_scores.argsort()[::-1][1:top_n+1]

    return df_cleaned.iloc[similar_song_indices][['track_name', 'artists', 'album_name', 'track_genre']]

# Function to fetch album artwork from Last.fm API (Use caching)
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
    
    return LOCAL_IMAGE_PATH  # ✅ Fallback to local image

# Streamlit UI
st.markdown('<div class="title-container"><h1>🎵 Music Recommender System</h1><p>Find songs similar to your favorites</p></div>', unsafe_allow_html=True)

# Dropdown with "Select a Song" option
song_options = ["Select a Song"] + df_cleaned["song_display"].unique().tolist()
song_selection = st.selectbox("🎶 Select a Song:", song_options, index=song_options.index("Select a Song"))

# Only proceed if a valid song is selected
if song_selection != "Select a Song":
    selected_song = df_cleaned[df_cleaned["song_display"] == song_selection]["track_name"].values[0]
    selected_artist = df_cleaned[df_cleaned["song_display"] == song_selection]["artists"].values[0]

    if st.button("🔍 Get Recommendations"):
        with st.spinner("🔄 Fetching recommendations..."):
            recommendations = content_based_recommend(selected_song)

            if recommendations is None or recommendations.empty:
                st.error(f"'{selected_song}' not found in the dataset. Please enter a valid song.")
            else:
                st.markdown('<div class="recommend-title">🎼 Recommended Songs</div>', unsafe_allow_html=True)

                # Display first 5 recommendations
                cols1 = st.columns(5)
                for i in range(5):
                    row = recommendations.iloc[i]
                    album_art_url = get_album_art(row['track_name'], row['artists'])
                    with cols1[i]:
                        st.image(album_art_url, width=150)
                        st.markdown(f"""
                            <div class="song-card">
                                <h3>{row['track_name']}</h3>
                                <h4>{row['artists']}</h4>
                                <p>{row['album_name']}</p>
                                <p><i>{row['track_genre']}</i></p>
                            </div>
                        """, unsafe_allow_html=True)

                # ✅ Space between first 5 and last 5
                st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

                # ✅ Display last 5 recommendations
                cols2 = st.columns(5)
                for i in range(5, 10):
                    row = recommendations.iloc[i]
                    album_art_url = get_album_art(row['track_name'], row['artists'])
                    with cols2[i - 5]:
                        st.image(album_art_url, width=150)
                        st.markdown(f"""
                            <div class="song-card">
                                <h3>{row['track_name']}</h3>
                                <h4>{row['artists']}</h4>
                                <p>{row['album_name']}</p>
                                <p><i>{row['track_genre']}</i></p>
                            </div>
                        """, unsafe_allow_html=True)

# Footer UI
st.markdown("""
    <div class="footer">
        🎵 Made with ❤️ by <a href="https://github.com/yourgithub" target="_blank">Your Name</a> | 
        Follow us on <a href="https://twitter.com/yourprofile" target="_blank">Twitter</a> |
        Contact: <a href="mailto:your-email@example.com">your-email@example.com</a>
    </div>
""", unsafe_allow_html=True)
