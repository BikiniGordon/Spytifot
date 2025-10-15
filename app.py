from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import json
from music_database_analyzer import find_songs_closest_to_playlist_average, reorder_playlist_for_flow, convert_key_to_scalar
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'music-playlist-secret-key-2024'

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the database
df_clean = None

def initialize_app():
    global df_clean
    
    csv_file = 'duuzu_song_database_cleaned.csv'
    if os.path.exists(csv_file):
        try:
            df_clean = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            try:
                df_clean = pd.read_csv(csv_file, encoding='latin1')
            except UnicodeDecodeError:
                df_clean = pd.read_csv(csv_file, encoding='cp1252')
        # print(f"Loaded {len(df_clean)} songs from database")
        
        if 'Source' not in df_clean.columns:
            df_clean['Source'] = 'Original Database'

        load_user_songs_to_database()
        load_persistent_playlists()
        return True
    else:
        # print("Database file not found. Please run the data processing first.")
        return False

def load_user_songs_to_database():
    global df_clean
    
    user_songs_file = 'user_added_songs.csv'
    if os.path.exists(user_songs_file):
        try:
            df_user = pd.read_csv(user_songs_file)
            if not df_user.empty:
                if 'Source' not in df_user.columns:
                    df_user['Source'] = 'Added by User'
                
                df_clean = pd.concat([df_clean, df_user], ignore_index=True) # Add user song to database
                # print(f"Loaded {len(df_user)} user-added songs")
        except Exception as e:
            print(f"Error loading user songs: {e}")

def load_persistent_playlists():
    playlists_dir = 'persistent_playlists'
    if os.path.exists(playlists_dir):
        # Load from JSON files
        playlist_files = [f for f in os.listdir(playlists_dir) if f.endswith('.json')]
        loaded_count = 0
        
        for playlist_file in playlist_files:
            try:
                playlist_path = os.path.join(playlists_dir, playlist_file)
                with open(playlist_path, 'r') as f:
                    playlist_data = json.load(f)
                # print(f"Found persistent playlist: {playlist_data['name']} ({playlist_data['song_count']} songs)")
                loaded_count += 1
            except Exception as e:
                print(f"Error loading playlist {playlist_file}: {e}")
        
        csv_files = [f for f in os.listdir(playlists_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(playlists_dir, csv_file)
            playlist_name = csv_file.rsplit('.', 1)[0].replace('_', ' ').title()
            
            # Check if already have a JSON for this playlist
            json_equivalent = csv_file.replace('.csv', '.json')
            # if json_equivalent not in playlist_files:
            #     print(f"Found unprocessed CSV playlist: {csv_file}")
                
        if loaded_count > 0:
            print(f"Total persistent playlists available: {loaded_count}")

def load_playlist_from_persistent_storage(playlist_name):
    playlists_dir = 'persistent_playlists'
    safe_filename = "".join(c for c in playlist_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename = safe_filename.replace(' ', '_') + '.json'
    playlist_file = os.path.join(playlists_dir, safe_filename)
    
    if os.path.exists(playlist_file):
        try:
            with open(playlist_file, 'r') as f:
                playlist_data = json.load(f)
            return playlist_data['songs']
        except Exception as e:
            print(f"Error loading playlist from storage: {e}")
            return None
    return None

@app.route('/api/playlists/load-persistent', methods=['GET'])
def get_persistent_playlists():
    playlists_dir = 'persistent_playlists'
    persistent_playlists = []
    
    if os.path.exists(playlists_dir):
        playlist_files = [f for f in os.listdir(playlists_dir) if f.endswith('.json')]
        
        for playlist_file in playlist_files:
            try:
                playlist_path = os.path.join(playlists_dir, playlist_file)
                with open(playlist_path, 'r') as f:
                    playlist_data = json.load(f)
                
                persistent_playlists.append({
                    'name': playlist_data['name'],
                    'song_count': playlist_data['song_count'],
                    'created_at': playlist_data.get('created_at', 'Unknown'),
                    'filename': playlist_file
                })
            except Exception as e:
                print(f"Error reading playlist {playlist_file}: {e}")
    
    return jsonify({
        'persistent_playlists': persistent_playlists,
        'total': len(persistent_playlists)
    })

@app.route('/api/playlists/load-persistent', methods=['POST'])
def load_persistent_playlist():
    data = request.get_json()
    playlist_name = data.get('name')
    
    if not playlist_name:
        return jsonify({'error': 'Playlist name is required'})
    
    playlist_songs = load_playlist_from_persistent_storage(playlist_name)
    
    if playlist_songs is None:
        return jsonify({'error': 'Playlist not found in persistent storage'})
    
    # Initialize session playlists
    if 'playlists' not in session:
        session['playlists'] = {}
    
    # Create unique name if already exists in session
    original_name = playlist_name
    counter = 1
    while playlist_name in session['playlists']:
        playlist_name = f"{original_name} (Loaded {counter})"
        counter += 1
    
    # Add to session
    session['playlists'][playlist_name] = playlist_songs
    session['current_playlist'] = playlist_name
    session['playlist'] = playlist_songs  # Backward compatibility
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Loaded playlist "{playlist_name}" with {len(playlist_songs)} songs',
        'playlist_name': playlist_name,
        'song_count': len(playlist_songs)
    })

def save_user_song_to_database(song_data):
    global df_clean
    
    user_songs_file = 'user_added_songs.csv'

    new_song_df = pd.DataFrame([{
        'ID': song_data['id'],
        'Artist': song_data['artist'],
        'Song': song_data['song'],
        'Key': song_data['key'],
        'BPM': song_data['bpm'],
        'Key_Scalar': song_data['key_scalar'],
        'Notes': song_data['notes'],
        'Source': 'Added by User'
    }])
    
    if os.path.exists(user_songs_file):
        try:
            df_existing = pd.read_csv(user_songs_file)
            # check for duplicate
            if not ((df_existing['Artist'] == song_data['artist']) & 
                   (df_existing['Song'] == song_data['song'])).any():
                df_combined = pd.concat([df_existing, new_song_df], ignore_index=True)
                df_combined.to_csv(user_songs_file, index=False)
                
                df_clean = pd.concat([df_clean, new_song_df], ignore_index=True)
                return True
        except Exception as e:
            print(f"Error updating user songs file: {e}")
            return False
    else:
        new_song_df.to_csv(user_songs_file, index=False) # new user songs file
        df_clean = pd.concat([df_clean, new_song_df], ignore_index=True)
        return True
    
    return False

def save_playlist_persistently(playlist_name, playlist_data):
    playlists_dir = 'persistent_playlists'
    os.makedirs(playlists_dir, exist_ok=True)
    
    safe_filename = "".join(c for c in playlist_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename = safe_filename.replace(' ', '_') + '.json'
    
    playlist_file = os.path.join(playlists_dir, safe_filename)
    
    try:
        playlist_info = {
            'name': playlist_name,
            'songs': playlist_data,
            'created_at': pd.Timestamp.now().isoformat(),
            'song_count': len(playlist_data)
        }
        
        with open(playlist_file, 'w') as f:
            json.dump(playlist_info, f, indent=2)
        
        # print(f"Saved playlist '{playlist_name}' persistently")
        return True
    except Exception as e:
        print(f"Error saving playlist: {e}")
        return False

@app.route('/api/search')
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a search term'})
    
    if df_clean is None:
        return jsonify({'error': 'Database not loaded'})
    
    # search songs
    query_lower = query.lower()
    artist_matches = df_clean[df_clean['Artist'].str.lower().str.contains(query_lower, na=False)]
    song_matches = df_clean[df_clean['Song'].str.lower().str.contains(query_lower, na=False)]
    
    # remove duplicate
    results = pd.concat([artist_matches, song_matches]).drop_duplicates()
    results = results.reset_index(drop=True)
    
    results = results.head(50) # limit to 50
    
    # convert to JSON
    songs = []
    for _, row in results.iterrows():
        songs.append({
            'id': int(row['ID']),
            'artist': row['Artist'],
            'song': row['Song'],
            'key': row['Key'],
            'bpm': int(row['BPM']),
            'key_scalar': int(row['Key_Scalar']),
            'source': row.get('Source', 'Original Database')
        })
    
    return jsonify({
        'songs': songs,
        'total': len(songs),
        'query': query
    })

@app.route('/api/playlist', methods=['GET'])
def get_playlist():
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
        session['current_playlist'] = 'Main Playlist'
        session.modified = True
    
    current_playlist_name = session.get('current_playlist', 'Main Playlist')
    
    # check playlist exists in playlists dict
    if current_playlist_name not in session['playlists']:
        session['playlists'][current_playlist_name] = []
        session.modified = True
    
    current_playlist = session['playlists'][current_playlist_name]
    
    # for backward compatibility
    session['playlist'] = current_playlist
    
    return jsonify({
        'playlist': current_playlist,
        'current_playlist': current_playlist_name
    })

@app.route('/api/playlist/add', methods=['POST'])
def add_to_playlist():
    data = request.get_json()
    song_id = data.get('song_id')
    
    if not song_id:
        return jsonify({'error': 'Song ID required'})
    
    if df_clean is None:
        return jsonify({'error': 'Database not loaded'})
    
    # song info
    song_row = df_clean[df_clean['ID'] == song_id]
    if song_row.empty:
        return jsonify({'error': 'Song not found'})
    
    # initialize playlists if not exists
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
        session['current_playlist'] = 'Main Playlist'
    
    current_playlist_name = session.get('current_playlist', 'Main Playlist')
    
    # check playlist exists in playlists dict
    if current_playlist_name not in session['playlists']:
        session['playlists'][current_playlist_name] = []
        session.modified = True
    
    current_playlist = session['playlists'][current_playlist_name]
    
    # check for duplicate
    if any(song['id'] == song_id for song in current_playlist):
        return jsonify({'error': 'Song already in playlist'})
    
    # add song to current playlist
    song_info = song_row.iloc[0]
    
    # required fields
    try:
        song_id = int(song_info['ID'])
        song_bpm = int(song_info['BPM'])
        song_key_scalar = int(song_info['Key_Scalar'])
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid song data: {str(e)}'})
    
    song_data = {
        'id': song_id,
        'artist': str(song_info['Artist']),
        'song': str(song_info['Song']),
        'key': str(song_info['Key']),
        'bpm': song_bpm,
        'key_scalar': song_key_scalar,
        'notes': str(song_info['Notes']) if pd.notna(song_info['Notes']) else '',
        'source': str(song_info.get('Source', 'Original Database'))
    }
    
    print(f"DEBUG: Adding song data: {song_data}")
    
    current_playlist.append(song_data)
    session['playlists'][current_playlist_name] = current_playlist
    # for backward compatibility
    session['playlist'] = current_playlist
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Added {song_info["Artist"]} - {song_info["Song"]} to {current_playlist_name}',
        'playlist_size': len(current_playlist),
        'added_song': int(song_info['ID']),
        'current_playlist': current_playlist_name
    })

@app.route('/api/playlist/remove', methods=['POST'])
def remove_from_playlist():
    data = request.get_json()
    song_id = data.get('song_id')
    
    if not song_id:
        return jsonify({'error': 'Song ID required'})
    
    # initialize playlists if not exists
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
        session['current_playlist'] = 'Main Playlist'
    
    current_playlist_name = session.get('current_playlist', 'Main Playlist')
    
    # ensure the current playlist exists
    if current_playlist_name not in session['playlists']:
        session['playlists'][current_playlist_name] = []
        session.modified = True
    
    current_playlist = session['playlists'][current_playlist_name]
    
    # remove song from current playlist
    original_length = len(current_playlist)
    current_playlist = [song for song in current_playlist if song['id'] != song_id]
    
    # update both storage locations
    session['playlists'][current_playlist_name] = current_playlist
    session['playlist'] = current_playlist  # backward compatibility
    session.modified = True
    
    if len(current_playlist) == original_length:
        return jsonify({'error': 'Song not found in playlist'})
    
    return jsonify({
        'success': True,
        'message': f'Song removed from {current_playlist_name}',
        'playlist_size': len(current_playlist),
        'current_playlist': current_playlist_name
    })

@app.route('/api/playlist/clear', methods=['POST'])
def clear_playlist():
    # initialize playlists if needed
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
        session['current_playlist'] = 'Main Playlist'
    
    current_playlist_name = session.get('current_playlist', 'Main Playlist')
    
    # clear the current playlist
    session['playlists'][current_playlist_name] = []
    session['playlist'] = []  # backward compatibility
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'{current_playlist_name} cleared',
        'current_playlist': current_playlist_name
    })

@app.route('/api/playlist/optimize', methods=['POST'])
def optimize_playlist():
    # initialize playlists if not exists
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
    if 'current_playlist' not in session:
        session['current_playlist'] = 'Main Playlist'
    
    current_playlist_name = session['current_playlist']
    current_playlist = session['playlists'].get(current_playlist_name, [])
    
    # if current playlist doesn't exist
    if current_playlist_name not in session['playlists']:
        session['playlists'][current_playlist_name] = []
        current_playlist = []
    
    if not current_playlist:
        return jsonify({'error': 'Playlist is empty'})
    
    if len(current_playlist) < 2:
        return jsonify({'error': 'Need at least 2 songs to optimize'})
    
    try:
        opts = {}
        try:
            opts = request.get_json(silent=True) or {}
        except Exception:
            opts = {}
        similarity_method = opts.get('similarity', 'euclidean')
        standardize_flag = bool(opts.get('standardize', False))
        # convert session playlist to format expected by music_database_analyzer
        # create a mapping to preserve source information
        playlist_data = []
        source_mapping = {}  # ID -> source
        print(f"DEBUG: Starting optimization with {len(current_playlist)} songs")
        
        for i, song in enumerate(current_playlist):
            print(f"DEBUG: Song {i+1}: {song}")
            song_id = song['id']

            source_mapping[song_id] = song.get('source', 'Original Database')
            
            playlist_data.append({
                'ID': song['id'],
                'Artist': song['artist'],
                'Song': song['song'],
                'Key': song['key'],
                'Key_Scalar': song['key_scalar'],
                'BPM': song['bpm'],
                'Notes': song.get('notes', '')
            })
        
        print(f"DEBUG: Converted playlist_data: {len(playlist_data)} songs")
        print(f"DEBUG: Source mapping: {source_mapping}")

        optimized_playlist, transition_scores = reorder_playlist_for_flow(playlist_data, similarity_method, standardize_flag)
        
        print(f"DEBUG: After optimization: {len(optimized_playlist)} songs")

        # convert back to session format, preserving source information
        optimized_session_playlist = []
        for song in optimized_playlist:
            song_id = song['ID']
            original_source = source_mapping.get(song_id, 'Original Database')
            
            optimized_session_playlist.append({
                'id': song['ID'],
                'artist': song['Artist'],
                'song': song['Song'],
                'key': song['Key'],
                'bpm': song['BPM'],
                'key_scalar': song['Key_Scalar'],
                'notes': song.get('Notes', ''),
                'source': original_source  # Preserve the original source
            })
        
        print(f"DEBUG: Final optimized playlist: {len(optimized_session_playlist)} songs with preserved sources")
        
        # multiple playlist system and backward compatibility
        session['playlists'][current_playlist_name] = optimized_session_playlist
        session['playlist'] = optimized_session_playlist
        session.modified = True
        
        avg_score = sum(transition_scores) / len(transition_scores) if transition_scores else 0
        
        return jsonify({
            'success': True,
            'message': f'Playlist optimized! Average similarity: {avg_score:.4f} (method: {similarity_method}, standardize: {standardize_flag})',
            'avg_similarity': avg_score,
            'similarity_method': similarity_method,
            'standardize': standardize_flag,
            'playlist': session['playlist']
        })
    
    except Exception as e:
        return jsonify({'error': f'Optimization canceled'})

# playlist management routes
@app.route('/api/playlists', methods=['GET'])
def get_all_playlists():
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
        session['current_playlist'] = 'Main Playlist'
        session.modified = True
    
    return jsonify({
        'playlists': session['playlists'],
        'current_playlist': session.get('current_playlist', 'Main Playlist')
    })

@app.route('/api/playlists/create', methods=['POST'])
def create_playlist():
    data = request.get_json()
    playlist_name = data.get('name', '').strip()
    
    if not playlist_name:
        return jsonify({'error': 'Playlist name is required'})
    
    if 'playlists' not in session:
        session['playlists'] = {}
    
    if playlist_name in session['playlists']:
        return jsonify({'error': 'Playlist with this name already exists'})
    
    session['playlists'][playlist_name] = []
    session['current_playlist'] = playlist_name
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Created playlist "{playlist_name}"',
        'playlist_name': playlist_name
    })

@app.route('/api/playlists/switch', methods=['POST'])
def switch_playlist():
    data = request.get_json()
    playlist_name = data.get('name')
    
    if not playlist_name:
        return jsonify({'error': 'Playlist name is required'})
    
    if 'playlists' not in session:
        session['playlists'] = {'Main Playlist': []}
    
    if playlist_name not in session['playlists']:
        return jsonify({'error': 'Playlist not found'})
    
    session['current_playlist'] = playlist_name
    # backward compatibility
    session['playlist'] = session['playlists'][playlist_name]
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Switched to playlist "{playlist_name}"',
        'current_playlist': playlist_name,
        'playlist': session['playlist']
    })

@app.route('/api/playlists/rename', methods=['POST'])
def rename_playlist():
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name', '').strip()
    
    if not old_name or not new_name:
        return jsonify({'error': 'Both old and new names are required'})
    
    if 'playlists' not in session:
        return jsonify({'error': 'No playlists found'})
    
    if old_name not in session['playlists']:
        return jsonify({'error': 'Playlist not found'})
    
    if new_name in session['playlists']:
        return jsonify({'error': 'A playlist with this name already exists'})
    
    # rename playlist
    session['playlists'][new_name] = session['playlists'].pop(old_name)
    
    # update current playlist
    if session.get('current_playlist') == old_name:
        session['current_playlist'] = new_name
    
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Renamed playlist from "{old_name}" to "{new_name}"',
        'new_name': new_name
    })

@app.route('/api/playlists/delete', methods=['POST'])
def delete_playlist():
    data = request.get_json()
    playlist_name = data.get('name')
    
    if not playlist_name:
        return jsonify({'error': 'Playlist name is required'})
    
    if 'playlists' not in session:
        return jsonify({'error': 'No playlists found'})
    
    if playlist_name not in session['playlists']:
        return jsonify({'error': 'Playlist not found'})
    
    if len(session['playlists']) == 1:
        return jsonify({'error': 'Cannot delete the last playlist'})
    
    # delete playlist
    del session['playlists'][playlist_name]
    
    # switch to another playlist if deleted
    if session.get('current_playlist') == playlist_name:
        remaining_playlists = list(session['playlists'].keys())
        session['current_playlist'] = remaining_playlists[0]
        session['playlist'] = session['playlists'][remaining_playlists[0]]
    
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': f'Deleted playlist "{playlist_name}"',
        'current_playlist': session.get('current_playlist')
    })

@app.route('/api/playlists/export/<playlist_name>')
def export_playlist(playlist_name):
    if 'playlists' not in session or playlist_name not in session['playlists']:
        return jsonify({'error': 'Playlist not found'})
    
    playlist = session['playlists'][playlist_name]
    if not playlist:
        return jsonify({'error': 'Playlist is empty'})
    
    try:
        playlist_df = pd.DataFrame(playlist)
        csv_filename = f"{playlist_name.replace(' ', '_').lower()}_playlist.csv"
        csv_path = os.path.join('exports', csv_filename)
        
        # create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        playlist_df.to_csv(csv_path, index=False)
        
        return jsonify({
            'success': True,
            'message': f'Exported playlist "{playlist_name}" to {csv_filename}',
            'filename': csv_filename
        })
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'})

@app.route('/api/recommendations')
def get_recommendations():
    if 'playlist' not in session or not session['playlist']:
        return jsonify({'error': 'Playlist is empty'})
    
    try:
        # convert session playlist format by music_database_analyzer
        playlist_data = []
        for song in session['playlist']:
            playlist_data.append({
                'ID': song['id'],
                'Artist': song['artist'],
                'Song': song['song'],
                'Key': song['key'],
                'Key_Scalar': song['key_scalar'],
                'BPM': song['bpm'],
                'Notes': song.get('notes', '')
            })
        
        recommendations_df = find_songs_closest_to_playlist_average(playlist_data, df_clean, 10)
        
        if recommendations_df.empty:
            return jsonify({'error': 'No recommendations found'})
        
        # to JSON format
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append({
                'id': int(row['ID']),
                'artist': row['Artist'],
                'song': row['Song'],
                'key': row['Key'],
                'bpm': int(row['BPM']),
                'key_scalar': int(row['Key_Scalar']),
                'distance': float(row['Distance_to_Playlist_Avg']),
                'source': row.get('Source', 'Original Database')
            })
        
        return jsonify({
            'recommendations': recommendations,
            'total': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to get recommendations: {str(e)}'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload-playlist', methods=['POST'])
def upload_playlist():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'})
        
        # secure and create playlist name
        filename = secure_filename(file.filename)
        playlist_name = filename.rsplit('.', 1)[0].replace('_', ' ').title()
        
        persistent_csv_path = os.path.join('persistent_playlists', filename)
        os.makedirs('persistent_playlists', exist_ok=True)
        file.save(persistent_csv_path)
        print(f"DEBUG: Saved CSV file persistently to {persistent_csv_path}")
        
        # read CSV
        try:
            df_csv = pd.read_csv(persistent_csv_path)
            print(f"DEBUG: Read CSV with {len(df_csv)} rows and columns: {list(df_csv.columns)}")
        except Exception as e:
            print(f"DEBUG: Error reading CSV: {e}")
            if os.path.exists(persistent_csv_path):
                os.remove(persistent_csv_path)
            return jsonify({'error': f'Failed to read CSV file: {str(e)}'})
        
        # validate required columns
        required_columns = ['Artist', 'Song']
        missing_columns = [col for col in required_columns if col not in df_csv.columns]
        
        if missing_columns:
            print(f"DEBUG: Missing columns: {missing_columns}")
            os.remove(persistent_csv_path)
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'})
        
        # initialize playlists if not exists
        if 'playlists' not in session:
            session['playlists'] = {'Main Playlist': []}
            session['current_playlist'] = 'Main Playlist'
        
        # unique playlist name
        original_name = playlist_name
        counter = 1
        while playlist_name in session['playlists']:
            playlist_name = f"{original_name} ({counter})"
            counter += 1
        
        print(f"DEBUG: Creating playlist: {playlist_name}")
        
        # Process each row in the CSV and create songs directly from CSV data
        playlist_songs = []
        processed_count = 0
        invalid_songs = []
        songs_added_to_db = 0
        
        # Get the current max ID from the database to generate new IDs
        max_id = int(df_clean['ID'].max()) if df_clean is not None else 0
        print(f"DEBUG: Max ID from database: {max_id}")
        
        for idx, row in df_csv.iterrows():
            try:
                artist = str(row['Artist']).strip()
                song = str(row['Song']).strip()
                
                if not artist or not song or artist == 'nan' or song == 'nan':
                    invalid_songs.append(f"Row {idx + 1}: Missing Artist or Song")
                    continue
                
                # Extract optional fields from CSV or use defaults
                key = str(row.get('Camelot', '0X')).strip() if pd.notna(row.get('Camelot')) else '0X'
                bpm = row.get('BPM', 120)
                key_scalar = convert_key_to_scalar(key)
                notes = str(row.get('Notes', '')).strip() if pd.notna(row.get('Notes')) else ''
                
                # Validate and convert BPM
                try:
                    bpm = int(float(bpm))
                    if bpm < 60 or bpm > 200:  # Reasonable BPM range
                        bpm = 120  # Default BPM
                except (ValueError, TypeError):
                    bpm = 120
                
                # Create song data with new ID
                song_id = int(max_id + idx + 1)
                song_data = {
                    'id': song_id,  # Ensure Python int, not numpy int64
                    'artist': str(artist),
                    'song': str(song),
                    'key': str(key),
                    'bpm': int(bpm),  # Ensure Python int
                    'key_scalar': int(key_scalar),  # Ensure Python int
                    'notes': str("Imported from CSV: " + notes),
                    'source': 'Added by User'
                }
                
                # Save song to database if it doesn't already exist
                if not ((df_clean['Artist'] == artist) & (df_clean['Song'] == song)).any():
                    if save_user_song_to_database(song_data):
                        songs_added_to_db += 1
                        print(f"Added to database: {artist} - {song}")
                
                playlist_songs.append(song_data)
                processed_count += 1
                
            except Exception as e:
                print(f"DEBUG: Error processing row {idx}: {e}")
                invalid_songs.append(f"Row {idx + 1}: {str(e)}")
                continue
        
        print(f"DEBUG: Processed {processed_count} songs, {len(invalid_songs)} invalid, {songs_added_to_db} added to database")
        
        # Create the new playlist
        session['playlists'][playlist_name] = playlist_songs
        session['current_playlist'] = playlist_name
        session['playlist'] = playlist_songs  # Backward compatibility
        session.modified = True
        
        # Save playlist persistently
        save_playlist_persistently(playlist_name, playlist_songs)
        
        response_data = {
            'success': True,
            'message': f'Created playlist "{playlist_name}" with {processed_count} songs from CSV. {songs_added_to_db} new songs added to database.',
            'playlist_name': playlist_name,
            'songs_created': processed_count,
            'songs_added_to_db': songs_added_to_db,
            'total_rows': len(df_csv),
            'csv_saved_to': persistent_csv_path
        }
        
        if invalid_songs:
            response_data['invalid_songs'] = invalid_songs[:10]  # Limit to first 10
            response_data['invalid_count'] = len(invalid_songs)
            if len(invalid_songs) > 10:
                response_data['message'] += f' ({len(invalid_songs)} rows had issues)'
        
        print(f"DEBUG: Returning success response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"DEBUG: Major error in upload_playlist: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up file if it exists
        if 'persistent_csv_path' in locals() and os.path.exists(persistent_csv_path):
            os.remove(persistent_csv_path)
        
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/api/stats')
def get_stats():
    if df_clean is None:
        return jsonify({'error': 'Database not loaded'})
    
    bpm_numeric = pd.to_numeric(df_clean['BPM'], errors='coerce')
    
    stats = {
        'total_songs': len(df_clean),
        'unique_keys': df_clean['Key'].nunique(),
        'bpm_range': {
            'min': int(bpm_numeric.min()),
            'max': int(bpm_numeric.max()),
            'avg': float(bpm_numeric.mean())
        },
        'key_distribution': df_clean['Key'].value_counts().head(10).to_dict()
    }
    
    return jsonify(stats)

@app.route('/')
def root():
    # redirect to search by default
    return redirect(url_for('search_page'))

@app.route('/search')
def search_page():
    # pass active identifier so sidebar can highlight
    return render_template('search.html', active='search')

@app.route('/playlist')
def playlist_page():
    return render_template('playlist.html', active='playlist')

@app.route('/library')
def library_page():
    return render_template('library.html', active='library')

if __name__ == '__main__':
    if initialize_app():
        print("Starting Music Playlist Web App...")
        # print("Open http://localhost:5000 in your browser")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("Failed to initialize app. Make sure duuzu_song_database_cleaned.csv exists.")
