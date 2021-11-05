import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

spotify = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager)


def analyze_playlist(creator, playlist_id):

    # Create empty dataframe
    playlist_features_list = ["artist", "album", "track_name", "track_id",
                              "danceability", "energy", "key", "loudness", "mode", "speechiness",
                              "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]
    playlist_df = pd.DataFrame(columns=playlist_features_list)

    # Create empty dict
    playlist_features = {}

    # Loop through every track in the playlist, extract features and append the features to the playlist df
    playlist = spotify.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        # Get audio features
        audio_features = spotify.audio_features(
            playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]

        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index=[0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index=True)
        playlist_df.to_csv("test.csv", index=True)

    return playlist_df



