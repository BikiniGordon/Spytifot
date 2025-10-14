SPYTIFOT — Linear Algebra Playlist Optimizer
===========================================

Overview
--------
This repository implements a small web app that demonstrates how linear algebra can be applied to improve Spotify-like playlist experiences. The app analyzes songs' BPM and musical key and uses vector similarity, similarity matrices, and matrix operations to reorder playlists so transitions feel smoother.

Key ideas and algorithms
------------------------
- Represent each song as a numeric feature vector (BPM × weight, Key scalar).
- Optionally standardize/normalize feature vectors to balance scales.
- Compute pairwise similarity between songs (Euclidean-derived similarity, Cosine similarity, and weighted variants).
- Build a similarity matrix and use a greedy path algorithm to reorder the playlist maximizing local similarity (smooth transitions).
- Use linear algebra primitives (vector differences, norms, dot-products, matrix construction) for all core calculations.

Why this is useful
------------------
This project shows how linear algebra underpins practical problems in music engineering: by converting musical attributes into vectors and manipulating them with matrix operations you can build useful features like playlist reordering, average-song recommendations, and transition scoring for a better listening flow.

Dataset
-------
The project uses duuzu's song key & BPM database (v10) as the primary dataset. Source:
https://docs.google.com/document/d/1WcHNaTo6KHNG88yUWxrCULuwHPuQCGQ8UtItgzzK50Q/edit?tab=t.0

Files & structure
-----------------
- `app.py` — Flask web app and API endpoints for search, playlists, optimization, and persistence.
- `music_database_analyzer.py` — Core algorithms: feature construction, similarity computation, greedy reorder, recommendations.
- `duuzu_song_database_cleaned.csv` — Primary dataset (song rows with BPM, Key, Key_Scalar, etc.).
- `templates/` — HTML templates used by the Flask app (UI for search, playlist manager, library).

Installation
------------
1. Create a Python 3.10+ virtual environment and activate it.
2. Install dependencies:

   pip install -r requirements.txt

3. Ensure the dataset `duuzu_song_database_cleaned.csv` is present in the repo root.
4. Start the app:

   python app.py

5. Open http://127.0.0.1:5000 in your browser.

Usage
-----
- Search songs and add them to your session playlist.
- Use "Optimize Flow" to reorder songs using Euclidean or Cosine similarity (normalization optional).

License & attribution
---------------------
This project is provided for educational purposes. The dataset is attributed to duuzu (link above). Please follow any usage terms in the original dataset source.