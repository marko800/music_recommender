from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib


# load song library
df = pd.read_csv("data/data_cleaned.csv")





def find_filler(song):
    # given a song (by its index), use previously trained KNN model to return the 23 (random choice) nearest neighbors

    # get song's data, preprocess
    x = df[df.index==song].drop(columns=['track_name','track_artist', 'track_album_name','track_album_release_date','duration_ms'])
    x_scaled = MinMaxScaler().fit_transform(x) # little bit of data leakage...

    # load model and find neighbors
    model = joblib.load("knn_model.pkl")
    result = model.kneighbors(x,n_neighbors=23)

    return df.iloc[list(result[1][0])[1:]]


def party_score(df):

    # first load song library and add party_score variable
    data = pd.read_csv("data/data_cleaned.csv")
    # party_score is define as follows: take two-dimensional variable (x,y)
    x,y = data.energy,data.danceability
    # project onto line x=y
    x_p = (x + y) / 2
    # take length of the vector (x_p,x_p) as new variable
    z = pd.DataFrame(np.sqrt(2)*x_p)
    # and normalize
    party_scaler = MinMaxScaler()
    data["party_score"] = party_scaler.fit_transform(z)

    # now apply to df
    df["party_score"] = party_scaler.transform(pd.DataFrame(np.sqrt(2)*(df.energy + df.danceability) / 2))
    return df




def generate_playlist(songs_df,step_size):
    #

    # load song library with party_score variable
    df = party_score(pd.read_csv("data/data_cleaned.csv"))

    # compute party_score for songs_df
    songs_df = party_score(songs_df)
    track_list = songs_df[["party_score"]]

    #extract two minima (will be start and end song) and remove from track list
    min1_idx = track_list.idxmin()
    start_song = track_list[track_list.index == min1_idx[0]]
    track_list.drop(index = min1_idx[0], inplace=True)
    min2_idx = track_list.idxmin()
    end_song = track_list[track_list.index == min2_idx[0]]
    track_list.drop(index = min2_idx[0], inplace=True)

    # create bins, depending on step_size
    dividers = [np.array([x*np.pi/step_size,(x+1)*np.pi/step_size]) for x in range(step_size)]
    bins = [(np.sin(x-np.pi/2)+1)/2 for x in dividers]

    # fill bins with songs according to their party_score
    binned_songs = []
    for bin in bins:
        in_this_bin = []
        for i,x in zip(track_list.index,track_list.party_score):
            if bin[0] < x and x <= bin[1]:
                in_this_bin.append(i)
        binned_songs.append(in_this_bin)

    # set nr of loops through binned_songs in order to put every song on the playlist
    l = max([len(i) for i in binned_songs])

    # successively add songs from binned_songs until all bins are empty, adding filler songs if necessary
    used_fillers = set() # keep track of fillers
    playlist = [start_song.index[0]] # initialise playlist with start_song

    for iteration in range(l):
        for elt in binned_songs:
            if not elt: # case no song in bin, fill with song that is close to previous one
                filler_df = find_filler(playlist[-1])
                filler_df = party_score(filler_df)

                # exclude already used filler songs
                available_fillers = filler_df[~filler_df.index.isin(used_fillers)]

                if not available_fillers.empty:
                    # find the song with party_score closest to previous one
                    prev_party_score = df.loc[playlist[-1], "party_score"]
                    closest_index = available_fillers.party_score.sub(prev_party_score).abs().idxmin()

                    # add to playlist and mark as used
                    playlist.append(closest_index)
                    used_fillers.add(closest_index)
                else:
                    # no valid filler found
                    print(f"No valid unique filler found.")
            else:
                playlist.append(elt[0])
                elt.remove(elt[0])

    playlist.append(end_song.index[0]) # add end_song

    return playlist
