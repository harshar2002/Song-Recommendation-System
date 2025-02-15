# 🎵 Music Recommender System

This **Music Recommender System** is built using **Streamlit, Pandas, NumPy, and Scikit-learn**. It recommends songs based on their similarity using **cosine similarity** and fetches album artwork via the **Last.fm API**.

---

## 📌 Features

### 🎶 **Song-Based Recommendations**
- Suggests **similar songs** based on a selected track.
- Uses **cosine similarity** to compare song features.

### 🎨 **Album Art Fetching**
- Retrieves album covers from the **Last.fm API**.
- Uses a **default fallback image** if album art is unavailable.

### 🎮 **Interactive UI**
- Built with **Streamlit**, offering an easy-to-use web interface.
- Displays song recommendations **with album art and metadata**.

### ⚡ **Optimized Performance**
- Implements **Streamlit caching** for efficient data handling.
- Reduces repeated API calls to **Last.fm** for better speed.

---

## 📚 Project Structure

```
📦 Music-Recommender-System
├── app.py                      # Main Streamlit app
├── dataset.csv                  # Original dataset
├── df_cleaned.csv                # Cleaned dataset
├── df_scaled.npy                 # Scaled NumPy array for faster processing
├── Song_Recommendation_System.ipynb  # Jupyter Notebook with model implementation
├── requirements.txt              # Required dependencies
├── images/
│   ├── ui_home.png               # UI Screenshot - Home Page
│   ├── ui_recommendations.png    # UI Screenshot - Recommendations
└── README.md                     # Project documentation
```

---

## 🔹 **UI Overview**

### 📌 **Home Page**
When you launch the app, the **main interface** looks like this:

![Home UI](images/ui_home.png)

- The user can **select a song** from the dropdown list.
- Click on **"🔍 Get Recommendations"** to see similar songs.

---

### 📌 **Recommendation Results**
Once a song is selected, the system recommends similar tracks:

![Recommendations UI](images/ui_recommendations.png)

- The app displays **recommended songs** along with **album art**.
- If album art is missing, a **default placeholder** is shown.
- Users can explore different tracks and find new music easily.

---

## ⚙️ Installation

### **1⃣ Clone the repository**
```sh
git clone https://github.com/harshar2002/Song-Recommendation-System.git
cd Song-Recommendation-System
```

### **2⃣ Install dependencies**
```sh
pip install -r requirements.txt
```

### **3⃣ Run the application**
```sh
streamlit run app.py
```

---

## 🎯 How It Works

### **1⃣ Data Preprocessing**
- Standardized using **MinMaxScaler** or **StandardScaler** from `scikit-learn`.
- Stores processed data in **NumPy array format (`df_scaled.npy`)** for faster access.

### **2⃣ Feature Engineering**
- Extracts relevant **audio features** such as:
  - **Tempo**
  - **Danceability**
  - **Energy**
  - **Valence**
  - **Loudness**
  - **Acousticness**
  - **Instrumentalness**

### **3⃣ Recommendation System**
- Uses **Cosine Similarity** to compare song features.
- Selects **top-N closest songs** based on similarity score.

### **4⃣ Streamlit UI**
- Users **select a song** from the dropdown menu.
- The system **displays recommended songs** with album art.
- Uses **Last.fm API** to fetch album images.

---

## 🛠️ Configuration

### **🔑 Last.fm API Key**
- Update `LASTFM_API_KEY` in **`app.py`** with your **own Last.fm API key**.
```python
LASTFM_API_KEY = "your_api_key_here"
```

### **🌆 Local Fallback Image**
- If album art is **not found**, the system uses `images.jpg`.
- You can replace it with another local image.

---

## 🤝 Contributing

1. **Fork** the repository.
2. Create a **new branch** (`feature-branch`).
3. Commit your changes (`git commit -m "Added feature X"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a **Pull Request (PR)**.

---

## 📝 License

This project is **MIT Licensed**.

---

## 📩 Contact

- **👤 Author**: Harsha  
- 📧 **Email**: [harshagowda497@gmail.com](mailto:harshagowda497@gmail.com)  
- 🌐 **GitHub**: [harshar2002](https://github.com/harshar2002)  

🎵 **Made with ❤️ by [Harsha](https://github.com/harshar2002)**

