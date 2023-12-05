import random
import timeit
import keras.models
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def getRandomSearch():
    # A list of all characters that can be chosen.
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']

    # Gets a random character from the characters string.
    randomCharacter = characters[random.randint(0, 25)]

    # Places the wildcard character at the beginning, or both beginning and end, randomly.
    switcher = random.randint(0, 1)
    if switcher == 0:
        randomSearch = randomCharacter + '%'
    else:
        randomSearch = '%' + randomCharacter + '%'

    return randomSearch


class MusicMoodClassifierHeur:
    def __init__(self):
        self.cid = "3e52bae82225408496d4e7090f4fd59c"
        self.secret = "119f51888db246ed8d0899902e302756"
        self.client_credentials_manager = SpotifyClientCredentials(client_id=self.cid, client_secret=self.secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        #self.estimator = keras.models.load_model('ml/music_model.h5')
        self.estimator = keras.models.load_model('C:/Users/rasik/OneDrive/Documents/Emotion_recognition/Emotion-Recognition-based-on-CNN/music_classifier/model/music_model.h5')

    def getTracks(self, query, number):
        start = timeit.default_timer()

        # create empty lists where the results are going to be stored
        artist_name = []
        track_name = []
        popularity = []
        track_id = []

        for i in range(0, number, 10):
            track_results = self.sp.search(q=query, type='track', limit=10, offset=i)
            for i, t in enumerate(track_results['tracks']['items']):
                artist_name.append(t['artists'][0]['name'])
                track_name.append(t['name'])
                track_id.append(t['id'])
                popularity.append(t['popularity'])

        stop = timeit.default_timer()
        #print('Time to run this code (in seconds):', stop - start)
        df_tracks = pd.DataFrame(
            {'artist_name': artist_name, 'track_name': track_name, 'track_id': track_id, 'popularity': popularity})
        return df_tracks

    def getAudioFeatures(self, tracks):
        # again measuring the time
        start = timeit.default_timer()

        # empty list, batchsize and the counter for None results
        rows = []
        batchsize = 100
        none_counter = 0

        for i in range(0, len(tracks['track_id']), batchsize):
            batch = tracks['track_id'][i:i + batchsize]
            feature_results = self.sp.audio_features(batch)
            for i, t in enumerate(feature_results):
                if t == None:
                    none_counter = none_counter + 1
                else:
                    rows.append(t)

        #print('Number of tracks where no audio features were available:', none_counter)

        stop = timeit.default_timer()
        #print('Time to run this code (in seconds):', stop - start)
        df_audio_features = pd.DataFrame.from_dict(rows, orient='columns')
        return df_audio_features

    def select_tracks(self,emotion):
        print("selecting tracks for mood:"+emotion)
        emotion_to_mood = {
            'Angry': 0.2,
            'Disgust': 0.1,
            'Fear': 0.15,
            'Happy': 0.9,
            'Sad': 0.05,
            'Surprise': 0.5,  # This can be adjusted based on context
            'Neutral': 0.5
        }
        mood = emotion_to_mood[emotion]
        print("emotion to mood:"+str(mood))
        selected_tracks_uri = []
        tracks=self.getTracks(getRandomSearch(), 501)
        #random.shuffle(tracks)
        tracks_all_data = self.getAudioFeatures(tracks)
        for index, track_data in tracks_all_data.iterrows():
            try:
                if mood < 0.10:
                    if (0 <= track_data["valence"] <= (mood + 0.15)
                        and track_data["danceability"] <= (mood+8)
                        and track_data["energy"] <= (mood+10)):
                        selected_tracks_uri.append(track_data)
                elif 0.10 <= mood < 0.25:
                    if ((mood - 0.075) <= track_data["valence"] <= (mood + 0.075)
                        and track_data["danceability"] <= (mood+4)
                        and track_data["energy"] <= (mood+5)):
                        selected_tracks_uri.append(track_data)
                elif 0.25 <= mood < 0.50:
                    if ((mood - 0.05) <= track_data["valence"] <= (mood + 0.05)
                        and track_data["danceability"] <= (mood+1.75)
                        and track_data["energy"] <= (mood+1.75)):
                        selected_tracks_uri.append(track_data)
                elif 0.50 <= mood < 0.75:
                    if ((mood - 0.075) <= track_data["valence"] <= (mood + 0.075)
                        and track_data["danceability"] <= (mood/2.5)
                        and track_data["energy"] >= (mood/2)):
                        selected_tracks_uri.append(track_data)
                elif 0.75 <= mood < 0.90:
                    if ((mood - 0.075) <= track_data["valence"] <= (mood + 0.075)
                        and track_data["danceability"] <= (mood/2)
                        and track_data["energy"] >= (mood/1.75)):
                        selected_tracks_uri.append(track_data)
                elif mood >= 0.90:
                    if ((mood - 0.15) <= track_data["valence"] < 1
                        and track_data["danceability"] <= (mood/1.75)
                        and track_data["energy"] >= (mood/1.5)):
                        selected_tracks_uri.append(track_data)
            except TypeError as te:
                print(te)
                continue

        response_results=[]
        for item in selected_tracks_uri:
            result = self.sp.track(item['id'])
            openLink="https://open.spotify.com/track/"+result['id']
            image =result['album']['images'][0]['url']
            name =result['name']
            response_results.append([name,openLink,image])

        return response_results[:5]
    

def get_music_from_func(emotion,improve_mood=True):
    mood_classifier = MusicMoodClassifierHeur()
    label = ''
    if improve_mood:
        if emotion:
            if(emotion == "Angry"):
                label = "Happy" # if angry detected, suggest calm music.
            elif(emotion == "Disgust"):
                label = "Happy"  # if disgusted detected, suggest calm music.
            elif(emotion == "Fear"):
                label = "Happy"  # if fearful detected, suggest calm music.
            elif(emotion == "Happy"):
                label = "Happy" # if happy detected, suggest happy music.
            elif(emotion == "Neutral"):
                label = "Happy" # if neutral detected, suggest happy music.
            elif(emotion == "Sad"):
                label = "Happy" # if sad detected, suggest sad music.
            elif(emotion == "Surprise"):
                label = "Happy" # if surprised detected, suggest energetic music.
        else:
            label =None
    else:
        label = emotion

    return mood_classifier.select_tracks(label)


if __name__=='__main__':
    print(get_music_from_func("Happy"))


"""
To suggest a mood value for each of these emotions in the context of music classification, 
we would typically use a scale from 0 to 1, where 0 represents the least positive or
 most negative emotion, and 1 represents the most positive emotion.
   The 'valence' in music terminology often reflects the musical positiveness conveyed by a track.
     However, assigning numerical mood values to these emotions can be somewhat subjective, 
     as the interpretation of music can vary widely among listeners.

Here's a suggested mapping of the emotions to mood values on a scale of 0 to 1:

Angry: Low valence, high energy. Suggested mood value: 0.2
Disgust: Low valence. Suggested mood value: 0.1
Fearful: Low valence, varying energy depending on whether it’s a suspenseful fear or an intense fear. 
            Suggested mood value: 0.15
Happy: High valence, high energy. Suggested mood value: 0.9
Sad: Low valence, low energy. Suggested mood value: 0.05
Surprise: This could vary. A positive surprise might have high valence,
         while a negative surprise might have low valence. Suggested mood value: 0.5 (neutral,\
         because it can go either way)
Neutral: Middle valence, not particularly high or low energy. Suggested mood value: 0.5
"""