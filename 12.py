from pandas import read_csv, unique
# 필요한 라이브러리 로드
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/alsgur/Documents/카카오톡 받은 파일/WISDM.csv", header=0, comment=";")
Walking = df[df['activity'] == 'Walking'].head(24000).copy()

Jogging = df[df['activity'] == 'Jogging'].head(24000).copy()

Upstairs = df[df['activity'] == 'Upstairs'].head(24000).copy()

Downstairs = df[df['activity'] == 'Downstairs'].head(24000).copy()

Sitting = df[df['activity'] == 'Sitting'].head(48000).copy()

Standing = df[df['activity'] == 'Standing'].head(48000).copy()

import pandas as pd

balanced_data = pd.DataFrame()

balanced_data = pd.concat([balanced_data, Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])

balanced_data.shape

print(balanced_data)
df=pd.concat([balanced_data.loc[balanced_data['activity']=='Jogging'], balanced_data.loc[balanced_data['activity']=='Sitting']],axis=0)
print(df)
Jogging = df[df['activity'] == 'Jogging'].head(4500).copy()
Sitting = df[df['activity'] == 'Sitting'].head(4500).copy()
df=pd.concat([Jogging,Sitting],axis=0)
df.drop(['Unnamed: 0'], axis=1)
df.loc[df['activity'] == 'Sitting'] = 0
df.loc[df['activity'] == 'Jogging'] = 1
print(df)
df['activity'].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['activity']=le.fit_transform(df['activity'])

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)



print(X_train.shape, X_test.shape)

# Normal = 0 Fall = 1

normal = X_train[X_train['activity'] == 0]

y_train = normal['activity']

X_train_normal_train = normal.drop(['activity'], axis=1)

y_test = X_test['activity']

X_test = X_test.drop(['activity'], axis=1)

X_train_ft = X_train_normal_train.values

X_test = X_test.values

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard

input_dim = X_train.shape[1]-1
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 100
batch_size = 32
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train_ft, X_train_ft,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
plt.show()

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
plt.plot(error_df['reconstruction_error'])
plt.plot(error_df['true_class'])
plt.show()

LABELS = ["Normal", "AbNormal"]

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

threshold = 0.73


y_pred = [0 if e > threshold else 1 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

precision, recall, f1,_ = precision_recall_fscore_support(y_test,y_pred,average='binary')

print ('Accuracy Score :',accuracy_score(error_df.true_class, y_pred) )

print ('Precision :',precision )

print ('Recall :',recall )

print ('F1 :',f1 )