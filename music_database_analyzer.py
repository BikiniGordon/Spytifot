#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def convert_key_to_camelot(df):
    key_to_camelot = {
        # Major keys (B = major)
        'Cmaj': '8B', 'C#maj': '3B', 'Dbmaj': '3B',
        'Dmaj': '10B', 'D#maj': '5B', 'Ebmaj': '5B',
        'Emaj': '12B', 'Fmaj': '7B',
        'F#maj': '2B', 'Gbmaj': '2B',
        'Gmaj': '9B', 'G#maj': '4B', 'Abmaj': '4B',
        'Amaj': '11B', 'A#maj': '6B', 'Bbmaj': '6B',
        'Bmaj': '1B',
        
        # Minor keys (A = minor)
        'Amin': '8A', 'A#min': '3A', 'Bbmin': '3A',
        'Bmin': '10A', 'Cmin': '5A',
        'C#min': '12A', 'Dbmin': '12A',
        'Dmin': '7A', 'D#min': '2A', 'Ebmin': '2A',
        'Emin': '9A', 'Fmin': '4A',
        'F#min': '11A', 'Gbmin': '11A',
        'Gmin': '6A', 'G#min': '1A', 'Abmin': '1A',
        
        'other': '0X'  # Special case for unknown keys
    }
    
    return df['Key'].map(key_to_camelot)


def convert_key_to_scalar(camelot_key):
    camelot_to_scalar = {
        # Minor keys (A) - inner wheel (0-11)
        '1A': 0, '2A': 1, '3A': 2, '4A': 3, '5A': 4, '6A': 5,
        '7A': 6, '8A': 7, '9A': 8, '10A': 9, '11A': 10, '12A': 11,
        
        # Major keys (B) - outer wheel (12-23)
        '1B': 12, '2B': 13, '3B': 14, '4B': 15, '5B': 16, '6B': 17,
        '7B': 18, '8B': 19, '9B': 20, '10B': 21, '11B': 22, '12B': 23,
        
        '0X': 24  # Special case for unknown keys
    }
    
    return camelot_to_scalar.get(camelot_key, 24)


def get_harmonic_compatible_keys(camelot_key):
    if camelot_key == '0X' or not camelot_key:
        return []
    
    # Extract number and letter
    try:
        number = int(camelot_key[:-1])
        letter = camelot_key[-1]
    except (ValueError, IndexError):
        return []
    
    compatible = []
    
    # Rule 1: Same number, different letter (relative major/minor)
    opposite_letter = 'A' if letter == 'B' else 'B'
    compatible.append(f"{number}{opposite_letter}")
    
    # Rule 2: Adjacent numbers, same letter
    prev_num = 12 if number == 1 else number - 1
    next_num = 1 if number == 12 else number + 1
    compatible.append(f"{prev_num}{letter}")
    compatible.append(f"{next_num}{letter}")
    
    # Rule 3: Perfect fourth/fifth relationships (advanced mixing)
    # +7 or -7 on the wheel (but staying within 1-12)
    fourth_up = ((number + 6) % 12) + 1 if ((number + 6) % 12) != 0 else 12
    fourth_down = ((number - 8) % 12) + 1 if ((number - 8) % 12) != 0 else 12
    compatible.append(f"{fourth_up}{letter}")
    compatible.append(f"{fourth_down}{letter}")
    
    return compatible


def search_songs(query, df):
    if not query.strip():
        print("Please enter a search term.")
        return pd.DataFrame()
    
    query = query.lower().strip()
    
    # Search both Artist and Song
    artist_matches = df[df['Artist'].str.lower().str.contains(query, na=False)]
    song_matches = df[df['Song'].str.lower().str.contains(query, na=False)]
    
    # Combine results and remove duplicates
    results = pd.concat([artist_matches, song_matches]).drop_duplicates()
    results = results.reset_index(drop=True)
    
    if len(results) == 0:
        print(f"No songs found for '{query}'")
        return pd.DataFrame()
    
    print(f"Found {len(results)} songs matching '{query}':")
    print("-" * 80)
    
    return results[['ID', 'Artist', 'Song', 'Key', 'BPM']]


def analyze_playlist_similarity(playlist, use_standardization=True, similarity_method='euclidean', bpm_weight=10.0, harmonic_weight=2.0):
    if not playlist:
        print("Playlist is empty - add some songs first")
        return None
    
    if len(playlist) < 2:
        print("Need at least 2 songs in playlist for similarity analysis")
        return None
    
    song_vectors = []
    song_labels = []
    song_keys = []  # for harmonic compatibility
    
    for song in playlist:
        # feature selection 2D vector with weighted BPM and Key_Scalar
        weighted_bpm = song['BPM'] * bpm_weight
        vector = [weighted_bpm, song['Key_Scalar']]
        song_vectors.append(vector)
        song_labels.append(f"ID:{song['ID']} - {song['Artist'][:15]}...")
        song_keys.append(song['Key'])  # for harmonic compatibility
    
    features = np.array(song_vectors) # to numpy array
    
    if len(np.unique(features, axis=0)) == 1:
        print("All songs have identical BPM and Key - no optimization needed!")
        return playlist, [1.0] * (len(playlist) - 1)  
    
    if use_standardization:
        scaler = StandardScaler() # - mean / s.d.
        features = scaler.fit_transform(features)
        
        # standardization resulted in zero vectors (all identical)
        if np.allclose(features, 0):
            print("Songs too similar after standardization - keeping original order")
            return playlist, [0.9] * (len(playlist) - 1)

    # similarity matrix !!! MATRIX OPERATION !!!
    # 3 choices: 1. Cosine  2. Euclidean    3. Weighted_Cosine
    n_songs = len(song_vectors)
    similarity_matrix = np.zeros((n_songs, n_songs))
    
    for i in range(n_songs):
        for j in range(n_songs):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                A = features[i]
                B = features[j]

                base_similarity = 0.0
                
                if similarity_method == 'cosine':
                    norm_A = norm(A)
                    norm_B = norm(B)
                    
                    if norm_A == 0 or norm_B == 0:
                        base_similarity = 0.9 # handle identical vectors
                    else:
                        cosine = np.dot(A, B) / (norm_A * norm_B)
                        base_similarity = cosine
                
                elif similarity_method == 'euclidean': # sqrt((A[0] - B[0])^2 + (A[1] - B[1])^2)
                    distance = np.linalg.norm(A - B) # .linalg.norm = find distance between point
                    base_similarity = 1 / (1 + distance) # convert distance to similarity 
                
                elif similarity_method == 'weighted_cosine':
                    norm_A = norm(A)
                    norm_B = norm(B)
                    
                    if norm_A == 0 or norm_B == 0:
                        base_similarity = 0.9
                    else:
                        cosine = np.dot(A, B) / (norm_A * norm_B)
                        base_similarity = cosine ** 3 # exponential penalty to reduce high values
                
                # harmonic compatibility
                key_A = song_keys[i]
                key_B = song_keys[j]
                
                compatible_keys = get_harmonic_compatible_keys(key_A)
                is_harmonic = key_B in compatible_keys or key_A == key_B
                
                if is_harmonic:
                    final_similarity = base_similarity * harmonic_weight
                    final_similarity = min(final_similarity, 1.0) # cap at 1.0
                else:
                    final_similarity = base_similarity
                
                similarity_matrix[i][j] = final_similarity
    
    return similarity_matrix, song_labels, song_vectors


def greedy_reorder_from_matrix(similarity_matrix):
    n = len(similarity_matrix)
    visited = [False] * n
    order = []

    current = 0
    order.append(current)
    visited[current] = True
    
    # greedy pick the most unvisited
    for _ in range(n - 1):
        best_similarity = -1
        next_song = -1
        
        for j in range(n):
            if not visited[j] and similarity_matrix[current][j] > best_similarity:
                best_similarity = similarity_matrix[current][j]
                next_song = j
        
        if next_song != -1:
            order.append(next_song)
            visited[next_song] = True
            current = next_song
    
    return order


def reorder_playlist_for_flow(playlist, similarity_method='euclidean', use_standardization=False):
    if not playlist or len(playlist) < 2:
        print("Need at least 2 songs to reorder")
        return playlist, []
    result = analyze_playlist_similarity(playlist, use_standardization, similarity_method)
    if result is None:
        return playlist, []
    
    similarity_matrix, song_labels, song_vectors = result

    reordered_indices = greedy_reorder_from_matrix(similarity_matrix)

    reordered_playlist = [playlist[i] for i in reordered_indices]
    
    # calculate transition scores
    transition_scores = []
    for i in range(len(reordered_indices) - 1):
        current_idx = reordered_indices[i]
        next_idx = reordered_indices[i + 1]
        score = similarity_matrix[current_idx][next_idx]
        transition_scores.append(score)
    
    return reordered_playlist, transition_scores

# recommendation
def find_songs_closest_to_playlist_average(playlist, df_clean, top_n=5):
    if not playlist:
        print("Playlist is empty - cannot calculate averages")
        return pd.DataFrame()

    total_key = 0
    total_bpm = 0
    count_key = 0
    count_bpm = 0

    playlist_song_id = set()
    
    for song in playlist:
        if song['Key_Scalar'] is not None:
            total_key += song['Key_Scalar']
            count_key += 1
        if song['BPM'] != '':
            total_bpm += int(song['BPM'])
            count_bpm += 1
        playlist_song_id.add(song['ID'])

    if count_key == 0 or count_bpm == 0:
        print("Cannot calculate averages - insufficient data")
        return pd.DataFrame()
    
    avg_key_scalar = total_key / count_key
    avg_bpm = total_bpm / count_bpm
    
    print(f"Playlist Average: BPM={avg_bpm:.1f}, Key_Scalar={avg_key_scalar:.1f}")
    
    # target vector
    target_vector = np.array([avg_bpm, avg_key_scalar])
   
    distances = []
    
    for idx, row in df_clean.iterrows():
        if row['ID'] in playlist_song_id:
            continue  # skip already in playlist

        song_vector = np.array([float(row['BPM']), float(row['Key_Scalar'])])

        distance = np.linalg.norm(song_vector - target_vector) # euclidean distance
        distances.append((distance, idx))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[0])
    
    # Get top closest songs
    closest_indices = [idx for _, idx in distances[:top_n]]
    closest_songs = df_clean.loc[closest_indices].copy()

    closest_songs['Distance_to_Playlist_Avg'] = [dist for dist, _ in distances[:top_n]]
    
    return closest_songs[['ID', 'Artist', 'Song', 'Key', 'BPM', 'Key_Scalar', 'Distance_to_Playlist_Avg']]

def convert_camelot_to_scalar_single(camelot_key):
    # Map Camelot keys to scalar values based on wheel position
    camelot_to_scalar = {
        # Minor keys (A) - inner wheel (0-11)
        '1A': 0, '2A': 1, '3A': 2, '4A': 3, '5A': 4, '6A': 5,
        '7A': 6, '8A': 7, '9A': 8, '10A': 9, '11A': 10, '12A': 11,
        
        # Major keys (B) - outer wheel (12-23)
        '1B': 12, '2B': 13, '3B': 14, '4B': 15, '5B': 16, '6B': 17,
        '7B': 18, '8B': 19, '9B': 20, '10B': 21, '11B': 22, '12B': 23,
        
        '0X': 24  # Unknown keys
    }
    
    return camelot_to_scalar.get(str(camelot_key).strip(), 24)

def save_combined_database(df_combined, output_file="duuzu_song_database_cleaned.csv"):
    print(f"Saving combined database to {output_file}")
    df_combined.to_csv(output_file, index=False)
    print(f"Saved {len(df_combined)} songs to {output_file}")


class PlaylistManager:
    def __init__(self, df_clean):
        self.df_clean = df_clean
        self.playlist = []
    
    def add_songs(self, song_ids):
        if isinstance(song_ids, int):
            song_ids = [song_ids]
        
        for song_id in song_ids:
            song_row = self.df_clean[self.df_clean['ID'] == song_id]
            
            if song_row.empty:
                print(f"Song ID {song_id} not found!")
                continue
            
            if song_id in [song['ID'] for song in self.playlist]:
                print(f"Song ID {song_id} already in playlist")
                continue
            
            song_info = song_row.iloc[0]
            self.playlist.append({
                'ID': song_info['ID'],
                'Artist': song_info['Artist'],
                'Song': song_info['Song'],
                'Key': song_info['Key'],
                'Key_Scalar': song_info['Key_Scalar'],
                'BPM': song_info['BPM'],
                'Notes': song_info['Notes']
            })
            
            print(f"Added: ID {song_id} - {song_info['Artist']} - {song_info['Song']}")
    
    def show_playlist(self):
        if not self.playlist:
            print("Playlist is empty")
            return
        
        print(f"\nCURRENT PLAYLIST ({len(self.playlist)} songs):")
        print("=" * 80)
        
        for i, song in enumerate(self.playlist, 1):
            bpm_str = f"{song['BPM']} BPM" if song['BPM'] != '' else "- BPM"
            print(f"{i:2d}. ID:{song['ID']:4d} | {song['Artist']} - {song['Song']} | "
                  f"{song['Key']} ({song['Key_Scalar']}) | {bpm_str}")
    
    def optimize_playlist_flow(self, similarity_method='euclidean'):
        if len(self.playlist) < 2:
            print("Need at least 2 songs to optimize!")
            return
        
        self.playlist, scores = reorder_playlist_for_flow(self.playlist, similarity_method)
        
        print(f"Playlist optimized using {similarity_method} similarity!")
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average transition similarity: {avg_score:.4f}")
    
    def search_songs(self, query):
        return search_songs(query, self.df_clean)
    
    def get_recommendations(self, top_n=5):
        return find_songs_closest_to_playlist_average(self.playlist, self.df_clean, top_n)
    
    def save_updated_database(self, filename="duuzu_song_database_cleaned.csv"):
        save_combined_database(self.df_clean, filename)

    def export_playlist(self, filename="playlist.csv"):
        if not self.playlist:
            print("Playlist is empty - nothing to export!")
            return
        
        playlist_df = pd.DataFrame(self.playlist)
        playlist_df.to_csv(filename, index=False)
        print(f"Playlist exported to {filename} ({len(self.playlist)} songs)")
