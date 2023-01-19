#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans


# In[2]:
st.write('Spotify Recommendation System')

data_dict=pickle.load(open('data.pkl','rb'))
data=pd.DataFrame(data_dict)


# In[3]:


num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = data.select_dtypes(include=num_types)


# In[6]:


kmeans = KMeans(n_clusters=6)
features = kmeans.fit_predict(num)
data['feature']=features


# In[7]:


class SpotifyRecommender():
    def __init__(self, rec_data):
        #our class should understand which data to work with
        self.rec_data_ = rec_data
    
    #function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations(self, song_name, amount=20):
        distances = []
        #choosing the data for our song
        song = self.rec_data_[(self.rec_data_.track_name.str.lower() == song_name.lower())].head(1).values[0]
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.track_name.str.lower() != song_name.lower()]
        for r_song in (res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                #indeces of non-numerical columns
                if not col in [0,1]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
        #sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        columns = ['track_name']
        return res_data[columns][:amount]


# In[9]:


Recommender=SpotifyRecommender(data)


# In[8]:


class SpotifyRecommender_Artist():
    def __init__(self_1, rec_data_1):
        #our class should understand which data to work with
        self_1.rec_data__1 = rec_data_1
    
    
    #function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations_artist(self_1, artist_name_1, amount=20):
        distances = []
        #choosing the data for our song
        artist = self_1.rec_data__1[(self_1.rec_data__1.artists.str.lower() == artist_name_1.lower())].head(1).values[0]
        #dropping the data with our song
        res_data_1 = self_1.rec_data__1[self_1.rec_data__1.artists.str.lower() != artist_name_1.lower()]
        for r_artist in (res_data_1.values):
            dist = 0
            for col in np.arange(len(res_data_1.columns)):
                #indeces of non-numerical columns
                if not col in [0,1]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(artist[col]) - float(r_artist[col]))
            distances.append(dist)
        res_data_1['distance'] = distances
        #sorting our data to be ascending by 'distance' feature
        res_data_1 = res_data_1.sort_values('distance')
        columns = ['track_name']
        return res_data_1[columns][:amount]


# In[11]:


Recommeder_artists=SpotifyRecommender_Artist(data)


# In[12]:


name_dict=pickle.load(open('names.pkl','rb'))
names=pd.DataFrame(name_dict)
song_dict=pickle.load(open('song_names.pkl','rb'))
song=pd.DataFrame(song_dict)
artists_dict=pickle.load(open('artist_name.pkl','rb'))
artist=pd.DataFrame(artists_dict)


# In[13]:


#st.write('Spotify Recommendation System')


# In[15]:


selected_name=st.selectbox('Enter song or Artist name',
                          names['name'].values)


# In[18]:


if st.button('Recommend'):
    if selected_name in song.values:
        recommended_song=Recommender.get_recommendations(selected_name)
        hide_table_row_index="""
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """
        st.markdown(hide_table_row_index,unsafe_allow_html=True)
        st.table(recommended_song)
    else:
            recommneded_artist=Recommeder_artists.get_recommendations_artist(selected_name)
            hide_table_row_index="""
                            <style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>
                            """
            st.markdown(hide_table_row_index,unsafe_allow_html=True)
            st.table(recommneded_artist)


# In[ ]:




