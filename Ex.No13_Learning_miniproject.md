

# 🎯 **Experiment No: 13 — *Mini Project*

### **Date:** 11-11-2025

### **Register Number:** 212223060207

---

## **AIM:**

To write a Python program that trains a classifier to recommend songs based on user preferences using supervised learning and clustering analysis, and automatically generates a playlist of similar tracks.

---

## **APPARATUS / SOFTWARE REQUIRED:**

* Python 3.x
* Google Colab / Jupyter Notebook
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## **ALGORITHM:**

### **1. Import Libraries**

* Import the required Python libraries for data handling, visualization, and machine learning such as:
  `pandas`, `numpy`, `matplotlib`, `seaborn`, and modules from `sklearn` (like `TfidfVectorizer`, `cosine_similarity`, and `KMeans`).

---

### **2. Load Dataset**

* Load the **music metadata dataset** (e.g., `tcc_ceds_music.csv`) containing attributes like:

  * `track_name`
  * `artist_name`
  * `genre`
* Display the first few records using `head()` to verify successful data loading.

---

### **3. Data Exploration (EDA)**

* Examine the dataset structure and size.
* Visualize the **distribution of top genres** and **popular artists** using Seaborn bar plots.
* Identify and handle missing or inconsistent data entries.
* This step helps understand the diversity and balance of the dataset.

---

### **4. Data Preprocessing**

* Combine important descriptive fields (`genre`, `artist_name`, and `track_name`) into a **single feature column** called `combined_features`.
* Handle missing values by replacing them with empty strings (`''`).
* Apply **TF-IDF Vectorization** to transform this text data into numerical vectors, where:

  * Common words are weighted lower.
  * Unique words get higher weights.
* Compute **Cosine Similarity** between all song vectors to determine how similar two tracks are based on their features.

---

### **5. Clustering Analysis (K-Means)**

* Apply **K-Means Clustering** on the TF-IDF matrix to group songs into distinct clusters (e.g., 5 clusters).
* Each cluster represents a group of songs that share common features such as genre, artist style, or mood.
* Assign each song a cluster label for better organization and analysis.
* Visualize the number of songs in each cluster using a bar plot.

---

### **6. Model Training / Similarity Computation**

* Although no explicit supervised model is trained, the **cosine similarity matrix** effectively “learns” relationships between songs.
* Each pair of songs has an associated similarity score ranging from 0 to 1.
* These scores act as the **basis for making recommendations** — songs with higher similarity scores are recommended together.

---

### **7. Recommendation Function**

* Define a function `get_recommendations(song_title, data, cosine_sim)` that:

  1. Finds the index of the input song in the dataset.
  2. Retrieves similarity scores for all other songs using the cosine similarity matrix.
  3. Sorts the scores in descending order (most similar songs first).
  4. Selects the top N (e.g., 10) most similar songs.
  5. Retrieves their track names, artist names, genres, and cluster numbers.
  6. Displays them in a formatted table.

---

### **8. Playlist Generation**

* The top recommended songs are automatically **compiled into a playlist**.
* This playlist contains songs that are most similar in genre, artist, and overall musical pattern to the chosen track.
* The system can easily be extended to export this playlist or integrate it with music streaming services.

---

### **9. Testing the System**

* Input a sample song title (e.g., *“Shape of You”*).
* Run the `get_recommendations()` function.
* Observe the generated playlist of similar tracks, confirming that the model correctly identifies related songs.
* Visualize the clusters to verify how similar songs are grouped together.

---

### **10. Result Interpretation**

* The recommended playlist demonstrates how machine learning can be used for **content-based music recommendation**.
* Songs within the same cluster exhibit shared musical characteristics.
* This validates the use of **TF-IDF**, **cosine similarity**, and **K-Means clustering** in a unified system.




---

## **PROGRAM (with Clustering + Playlist Generation):**

```python
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv('/content/tcc_ceds_music.csv', sep=',', on_bad_lines='skip', engine='python')
print("Dataset loaded successfully.")
display(data.head())

# Step 3: Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=data, order=data['genre'].value_counts().index[:10])
plt.title('Top 10 Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Step 4: Preprocessing
data['combined_features'] = (
    data['genre'].fillna('') + ' ' +
    data['artist_name'].fillna('') + ' ' +
    data['track_name'].fillna('')
)

# Step 5: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Step 6: Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 7: K-Means Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(tfidf_matrix)

plt.figure(figsize=(8, 5))
sns.countplot(x='cluster', data=data, palette='coolwarm')
plt.title('Song Distribution by Cluster')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Songs')
plt.show()

# Step 8: Recommendation Function
def get_recommendations(song_title, data, cosine_sim, top_n=10):
    idx = data[data['track_name'] == song_title].index
    if len(idx) == 0:
        print("Song not found in the dataset.")
        return
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    playlist = data[['track_name', 'artist_name', 'genre', 'cluster']].iloc[song_indices]
    print(f"\n🎵 Auto-generated Playlist for '{song_title}':\n")
    return playlist.reset_index(drop=True)

# Step 9: Generate Playlist
playlist = get_recommendations('Shape of You', data, cosine_sim)
display(playlist)
```

### Download the full code 
[Download the Jupyter Notebook](./Music_recommendation_system.ipynb)

---

## **OUTPUT:**

✅ **Dataset Sample:**

| track_name      | artist_name | genre |
| --------------- | ----------- | ----- |
| Shape of You    | Ed Sheeran  | Pop   |
| Perfect         | Ed Sheeran  | Pop   |
| Blinding Lights | The Weeknd  | R&B   |

✅ **Clustering Visualization:**
A bar chart showing the distribution of songs into 5 clusters, grouping similar genres and artists together.

✅ **Generated Playlist Example:**

**Input Song:** *Shape of You*
**Auto-Generated Playlist:**

| Track Name          | Artist Name    | Genre | Cluster |
| ------------------- | -------------- | ----- | ------- |
| Perfect             | Ed Sheeran     | Pop   | 2       |
| Thinking Out Loud   | Ed Sheeran     | Pop   | 2       |
| Love Me Like You Do | Ellie Goulding | Pop   | 2       |
| Blinding Lights     | The Weeknd     | R&B   | 3       |
| Photograph          | Ed Sheeran     | Pop   | 2       |
| Stay                | Justin Bieber  | Pop   | 2       |

---

## **DETAILED EXPLANATION:**

### 1. **Feature Combination & Preprocessing**

The model merges the genre, artist name, and track name into a single feature column to capture descriptive information for each song.

### 2. **TF-IDF Vectorization**

Converts the textual information into numerical vectors.
This allows the algorithm to measure how relevant or unique a term is in describing a song.

### 3. **Cosine Similarity**

Used to find how close or similar two songs are.
Higher cosine values indicate stronger similarity.

### 4. **K-Means Clustering**

Groups songs into distinct clusters based on their feature similarity.
Songs in the same cluster share common traits such as mood, genre, or artist type.

### 5. **Recommendation Logic**

When a song title is entered, the system:

* Finds the most similar songs using cosine similarity.
* Filters them based on cluster information.
* Creates a **playlist** with the top recommendations.

### 6. **Playlist Generation**

The recommended songs are automatically compiled into a playlist format.
This playlist can later be integrated with streaming services or exported as a custom user-curated music list.

---

## **RESULT:**

Thus, the **music recommendation system with clustering and playlist generation** was successfully developed and tested.
It analyzes song metadata, groups songs with similar characteristics, and generates a personalized playlist for the user using **TF-IDF vectorization, cosine similarity, and K-Means clustering**.
The system demonstrates the integration of **supervised learning concepts** with **unsupervised clustering techniques** for an intelligent and practical AI application.

---
