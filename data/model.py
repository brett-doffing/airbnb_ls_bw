import numpy as np 
import pandas as pd
from keras import models, layers, optimizers, regularizers
from keras.utils.vis_utils import model_to_dot

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://raw.githubusercontent.com/leibo411/airbnb_ls_bw/main/data/listings3.csv')

def wrangle_bathrooms_text(x):
    try:
        # Split bathroom_text
        split_text = str(x).split()
        # Grab what is hopefully a float
        first_split = split_text[0]
        # if float assignment fails, see exception
        num_bathrooms = float(first_split)
    except ValueError:
        # These are all half-baths
        return 0.5
    else:
        return num_bathrooms

df['bathrooms'] = df.bathrooms_text.apply(wrangle_bathrooms_text)

# Function to load in data from the Airbnb hong kong
def load_data():
  # df = pd.read_csv('https://raw.githubusercontent.com/leibo411/airbnb_ls_bw/main/data/listings3.csv')
  df_number = df.select_dtypes(include='number')
  df_number = df_number.drop(columns=['id', 'scrape_id', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'license', 'availability_60', 'number_of_reviews_l30d', 
  'maximum_nights_avg_ntm', 'review_scores_communication', 'review_scores_location', 'availability_90', 'review_scores_checkin', 'calculated_host_listings_count_entire_homes', 
  'number_of_reviews_ltm', 'minimum_minimum_nights', 'maximum_minimum_nights','minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'review_scores_cleanliness', 
  'review_scores_rating', 'review_scores_accuracy', 'review_scores_value', 'calculated_host_listings_count', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 
  'reviews_per_month', 'host_id', 'host_listings_count'])
  df_number = df_number.fillna(method='ffill')
  df_host = df[['host_location', 'price']]
  df_num = pd.concat([df_host, df_number], axis=1)
  df_num['price'] = df_num['price'].str.replace('$', '').str.replace(',', '').astype(float)
  # df_num['price'].astype(float) 
  return df_num

df = load_data()

# print(df.head())

# identify the target for the model
X = np.array(df.drop(columns=['price', 'host_location'])) # might have to be a numpy array, original code was  X = np.array(df.drop(columns='Survived'))
y = df['price']

# Make a train test split
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.fit_transform(X_test1)

columns = ['host_total_listings_count', 'latitude', 'longitude', 'accommodates',
       'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_365',
       'number_of_reviews', 'bathrooms']

# Building the model
nn2 = models.Sequential()
nn2.add(layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(128, activation='relu'))
nn2.add(layers.Dense(16, activation='relu'))
nn2.add(layers.Dense(1, activation='linear'))

# Compiling the model
nn2.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_squared_error'])

# Printing the model summary
print(nn2.summary())


# Training the model
nn2_history = nn2.fit(X_train,
                  y_train,
                  epochs=1000,
                  batch_size=256,
                  validation_split = 0.1)

# dummy_data = {'host_total_listing_count': [1], 'latitude': [17.234], 'longitude': [17.234], 'accommodates': [4], 'bedrooms': [3], 'beds': [3], 
# 'minimum_nights': [2], 'maximum_nights': [90], 'availability_30': [20], 'availability_365': [300], 'number_of_reviews': [40], 'bathrooms': [1]}
# df_dummy = pd.DataFrame(data=dummy_data)

# df_dummy = pd.DataFrame(data=dummy_data, dtype=np.int64)

# X = np.asarray(df_dummy).astype(np.float64)

# nn2.predict(X)