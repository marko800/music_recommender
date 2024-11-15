from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

def find_filler(song):
    # Given a song (by its index), use previously trained KNN model to return the 23 (random choice) nearest neighbors

    # load dataframe
    df = pd.read_csv("data/data_cleaned.csv")

    # get song's data, preprocess
    x = df[df.index==song].drop(columns=['track_name','track_artist', 'track_album_name','track_album_release_date','duration_ms'])
    x_scaled = MinMaxScaler().fit_transform(x) # little bit of data leakage...

    # load model and find neighbors
    model = joblib.load("knn_model.pkl")
    result = model.kneighbors(x,n_neighbors=23)

    return df.iloc[list(result[1][0])[1:]]
