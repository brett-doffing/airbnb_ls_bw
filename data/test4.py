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

# Function to load in data from the Airbnb hong kong
def load_data():
  df = pd.read_csv('https://raw.githubusercontent.com/leibo411/airbnb_ls_bw/main/data/listings3.csv')
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
       'number_of_reviews']

# Building the model
nn2 = models.Sequential()
nn2.add(layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(512, activation='relu'))
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

# def nn_model_evaluation(model, skip_epochs=0, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
#     """
#     For a given neural network model that has already been fit, prints for the train and tests sets the MSE and r squared
#     values, a line graph of the loss in each epoch, and a scatterplot of predicted vs. actual values with a line
#     representing where predicted = actual values. Optionally, a value for skip_epoch can be provided, which skips that
#     number of epochs in the line graph of losses (useful in cases where the loss in the first epoch is orders of magnitude
#     larger than subsequent epochs). Training and test sets can also optionally be specified.
#     """

    # # MSE and r squared values
    # y_test_pred = model.predict(X_test)
    # y_train_pred = model.predict(X_train)
    # print("Training MSE:", round(mean_squared_error(y_train, y_train_pred),4))
    # print("Validation MSE:", round(mean_squared_error(y_test, y_test_pred),4))
    # print("\nTraining r2:", round(r2_score(y_train, y_train_pred),4))
    # print("Validation r2:", round(r2_score(y_test, y_test_pred),4))
    
    # # Line graph of losses
    # model_results = model.history.history
    # plt.plot(list(range((skip_epochs+1),len(model_results['loss'])+1)), model_results['loss'][skip_epochs:], label='Train')
    # plt.plot(list(range((skip_epochs+1),len(model_results['val_loss'])+1)), model_results['val_loss'][skip_epochs:], label='Test', color='green')
    # plt.legend()
    # plt.title('Training and test loss at each epoch', fontsize=14)
    # plt.show()
    
    # # Scatterplot of predicted vs. actual values
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # fig.suptitle('Predicted vs. actual values', fontsize=14, y=1)
    # plt.subplots_adjust(top=0.93, wspace=0)
    
    # ax1.scatter(y_test, y_test_pred, s=2, alpha=0.7)
    # ax1.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
    # ax1.set_title('Test set')
    # ax1.set_xlabel('Actual values')
    # ax1.set_ylabel('Predicted values')
    
    # ax2.scatter(y_train, y_train_pred, s=2, alpha=0.7)
    # ax2.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
    # ax2.set_title('Train set')
    # ax2.set_xlabel('Actual values')
    # ax2.set_ylabel('')
    # ax2.set_yticklabels(labels='')
    
    # plt.show()