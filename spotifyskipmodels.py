# -*- coding: utf-8 -*-
"""SpotifySkipModels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ULFZj0x8RyRB9x3xzqiZs5IqdpuMTlVc
"""

# @title Download Training Set and untar
#runtime about 3 minutes
!wget https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/split-files/training_set_0.tar.gz

!tar -xvf 'training_set_0.tar.gz'
#Untars the dataset 
#runtime about 4 minutes

# @title Convert into dataframe and merge skip columns 
import os
import pandas as pd
import glob
import pickle

# merge the skip columns into one: based on skip_3 values 
# is skip_3 is true -> skip is true
# if skip_3 is false -> skip is false
# merging the files
joined_files = os.path.join("./training_set", "log_0_*.csv")
  
# A list of all joined files is returned
joined_list = glob.glob(joined_files)

# Finally, the files are joined
data = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
#data = pd.concat(map(pd.read_csv, glob.glob(os.path.join("./training_set", "log_0_*.csv"))), ignore_index= True)

# @title Merge CSVs
CHUNK_SIZE = 50000
output_file = "output.csv"

CHUNK_SIZE = 50000

first_one = True
for csv_file_name in joined_list:

    if not first_one: # if it is not the first csv file then skip the header row (row 0) of that file
        skip_row = [0]
    else:
        skip_row = []

    chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE, skiprows = skip_row)
    for chunk in chunk_container:
        chunk.to_csv(output_file, mode="a", header=False, index=False)
    first_one = False

# @title Convert into dataframe and merge skip columns
output_file = "output.csv"
data = pd.read_csv(output_file, names=["session_id", "session_position", "session_length", "track_id_clean", "skip_1", "skip_2", "skip_3", "not_skipped", "context_switch", "no_pause_before_play", "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback", "hist_user_behavior_is_shuffle", "hour_of_day", "date", "premium", "context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end"], low_memory=False)
data

# aggregate skip columns: based on skip_3 values 
# is skip_3 is true -> skip is true
# if skip_3 is false -> skip is false

data = data[~data["not_skipped"].isnull()]
data = data[~data["hist_user_behavior_reason_start"].isnull()]
data["skipped"] = data["skip_1"] | data["skip_2"] | data["skip_3"]
# normalize columns so that they are float vlaues 
data["skipped"] = data["skipped"].astype(int)
data['not_skipped'] = data['not_skipped'].astype(int)
data["session_length"] = (data["session_length"].astype(int)) / 20
data['hist_user_behavior_is_shuffle'] = data['hist_user_behavior_is_shuffle'].astype(int)
# catalog, radio, editorial_playlist, user_collection, personalized_playlist, and charts -- normalized from 0 to 5
data.loc[data['context_type'] == 'catalog', 'context_type'] = 0.0
data.loc[data['context_type'] == 'radio', 'context_type'] = .2
data.loc[data['context_type'] == 'editorial_playlist', 'context_type'] = .4
data.loc[data['context_type'] == 'user_collection', 'context_type'] = .6
data.loc[data['context_type'] == 'personalized_playlist', 'context_type'] = .8
data.loc[data['context_type'] == 'charts', 'context_type'] = 1.0
del data['skip_1']
del data['skip_2']
del data['skip_3']

for col in data.columns:
    print(col)

data.head()

# @title Selecting the features to extract
# skip_1: context_type, session_length, hist_user_behavior_n_seekback, hist_user_behavior_is_shuffle, context_switch, no_pause_before_play, session_position, long_pause_before_play
# skip_2: session_length, hist_user_behavior_n_seekfwd, hist_user_behavior_is_shuffle, context_switch, no_pause_before_play, long_pause_before_play
# skip_3: session_length, hist_user_behavior_n_seekfwd, no_pause_before_play, long_pause_before_play

# overall skip: context_type, session_length, hist_user_behavior_n_seekback, hist_user_behavior_n_seekfwd, hist_user_behavior_is_shuffle, context_switch, no_pause_before_play, session_position, long_pause_before_play
# extract features from skip_1, skip_2, and skip_3
# feed into classifier the training set values of the extracted features for skip_2 and skip_3 and not_skipped
# then, use random forest classifier to predict whether a given session is part of which 2 categories (skip_1/skip_2/skip_3, or not_skipped)
# evaluate based on how accurate it was in predicting for the test set
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

x = data[['session_length', 'context_type', 'hist_user_behavior_n_seekback', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_is_shuffle', 'context_switch', 'no_pause_before_play', 'session_position', 'long_pause_before_play']]  # Features
y = data['skipped']  # Labels

# @title Define train and test variables

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

clf=RandomForestClassifier(n_estimators=100,max_depth=5,verbose=2)

#Train the model
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test,y_pred))

# @title Exploratory Data Analysis: Feature Importance
feature_names = ['session_length', 'context_type', 'hist_user_behavior_n_seekback', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_is_shuffle', 'context_switch', 'no_pause_before_play', 'session_position', 'long_pause_before_play']
feature_imp = pd.Series(clf.feature_importances_,index=['Session Length', 'Context Type', 'User Behavior Seek Forward', 'User Behavior Seek Backward', 'User Behavior Shuffle', 'Context Switch', 'No Pause Before Play', 'Session Position', 'Long Pause Before Play']).sort_values(ascending=False)
feature_imp

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# @title Gradient Boosted Tree
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

x = data[['session_length', 'context_type', 'hist_user_behavior_n_seekback', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_is_shuffle', 'context_switch', 'no_pause_before_play', 'session_position', 'long_pause_before_play']]  # Features
y = data['skipped']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

gradient_booster = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, verbose=1)

gradient_booster.fit(x_train,y_train)
y_pred=gradient_booster.predict(x_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))

# @title LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, Dropout, Dense, LSTM, Embedding, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import LambdaCallback
from sklearn.preprocessing import MinMaxScaler
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import sys
from keras.utils.np_utils import to_categorical

embed_dim = 128
lstm_out = 100
batch_size = 300

x = data[['session_length', 'context_type', 'hist_user_behavior_n_seekback', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_is_shuffle', 'context_switch', 'no_pause_before_play', 'session_position', 'long_pause_before_play']]  # Features
y = data['skipped']
x = np.asarray(x)
y = np.asarray(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 36)
labels = 9
y_train = tf.one_hot(y_train, depth=labels)
y_train = np.asarray(y_train).reshape((-1,1))
x_train = x_train.reshape((-1,1))
print(y_train.shape)
print(x_train.shape)
model=tf.keras.Sequential([
 # add an embedding layer
 tf.keras.layers.Embedding(2500, embed_dim,input_length = x_train.shape[1]),
 tf.keras.layers.Dropout(0.2),
 # add a lstm layer
 tf.keras.layers.LSTM(lstm_out),
 # add flatten
 #tf.keras.layers.Flatten(),  #<========================
 # add a dense layer
 tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(32, activation=tf.keras.activations.softmax),
 # add the prediction layer
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax),
])

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#Here we fit the model
sc = model.fit(x_train.astype('float32'), y_train.astype('float32'), batch_size = 300, epochs = 100,  verbose = 5)

#Evaluate model and calculate accuracy/loss
#y_test = tf.one_hot(y_test, depth=labels)
y_test = np.asarray(y_test)
print(x_test.shape)
print(y_test.shape)
score,acc = model.evaluate(x_test.astype('float32'), y_test.astype('float32'), verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))

plt.plot(sc.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()