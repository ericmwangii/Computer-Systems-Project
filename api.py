from collections import defaultdict
import os
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from flask import Flask, request, redirect, url_for, render_template


spotify_data = pd.read_csv('./data/data.csv')


CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET))


song_cluster_pipeline = Pipeline(
    [('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20))])

X = spotify_data.select_dtypes(np.number)

number_cols = list(X.columns)
song_cluster_pipeline.fit(X)


def find_song(name):

    song_data = defaultdict()
    results = sp.search(q='track: {} '.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]

    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    # song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    # print(song_data)
    return pd.DataFrame(song_data)


number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_average_vector(song):
    song_vectors = []
    song_data = find_song(song)
    song_vector = song_data[number_cols].values
    song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    # print(song_matrix)
    return np.mean(song_matrix, axis=0)


def reccomend_songs(song, n_songs=10):
    metadata_cols = ['name', 'artists']
    song_center = get_average_vector(song)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    # rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    # print(str(rec_songs))
    return rec_songs[metadata_cols]


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        name = request.form['recommend']
        return redirect(url_for('success', name=name))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(reccomend_songs(name)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)

# reccomend_songs({'name': 'Faucet'})
