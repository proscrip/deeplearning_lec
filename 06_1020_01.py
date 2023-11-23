import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data_set = pd.read_csv('C:/Users/alsgur/Downloads/train.csv')
data_set['종목코드'] = label.fit_transform(data_set['종목코드'])

def serch(code, start, end):
     return data_set[(data_set['종목코드']==code)&(data_set['일자']>=start)&(data_set['일자']<=end)]

df = serch(10, 20210601, 20230530)

df_target = pd.DataFrame(data=df, columns=['종목코드'])
df_target = pd.get_dummies(df_target['종목코드'])
df_data = df.drop(columns=['종목명'])
print(df_data.columns)

def normalize_data(dataset):
    cols = dataset.columns.tolist()
    col_name = [0] * len(cols)
    for i in range(len(cols)):
        col_name[i] = i
    dataset.columns = col_name
    dtypes = dataset.dtypes.tolist()
    minmax = list()
    for column in dataset:
        dataset = dataset.astype({column: 'float32'})
    for i in range(len(cols)):
        col_values = dataset[col_name[i]]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for column in dataset:
        values = dataset[column].values
        for i in range(len(values)):
            values[i] = (values[i] - minmax[column][0]) / (minmax[column][1] - minmax[column][0])
        dataset[column] = values
    dataset[column] = values
    return dataset, minmax


dataset, minmax = normalize_data(df_data)
print(df.values)
values = dataset.values


def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def data_setup(n_steps, n_seq, sequence):
    X, y = split_sequences(sequence, n_steps)
    n_features = X.shape[2]
    X = X.reshape((len(X), n_steps, n_features))
    new_y = []
    for term in y:
        new_term = term[-1]
        new_y.append(new_term)
    return X, np.array(new_y), n_features


n_steps = 10
n_seq = 10000
rel_test_len = 0.1
X, y, n_features = data_setup(n_steps, n_seq, values)
X = X[:-1]
y = y[1:]
X_test, y_test = X[:int(len(X) * rel_test_len)], y[:int(len(X) * rel_test_len)]
X_train, y_train = X[int(len(X) * rel_test_len):], y[int(len(X) * rel_test_len):]
X.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(64, activation=None, input_shape=(10,4), return_sequences = True))
model.add(LSTM(32, activation=None, return_sequences = True))
model.add(Flatten())
model.add(Dense(100, activation=None))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
model = Sequential()
model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(10,4)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

import os
from keras import callbacks
epochs = 5000
verbosity = 2
dirx = '/.'
os.chdir(dirx)
h5 = 'network.h5'
checkpoint = callbacks.ModelCheckpoint(h5,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1)
callback = [checkpoint]
json = 'network.json'
model_json = model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=len(X_train) // 4,
                    validation_data = (X_test,y_test),
                    verbose=verbosity,
                    callbacks=callback)

from keras.models import load_model, model_from_json
def load_keras_model(optimizer):
    dirx = 'XXXXXXX'
    os.chdir(dirx)
    json_file = open('Convnet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=optimizer, loss='mse')
    model.load_weights('Convnet.h5')
    return model
model = load_keras_model('adam')

model.evaluate(X_test,y_test)

from matplotlib import pyplot as plt
pred_test = model.predict(X_test)
plt.plot(pred_test,'r')
plt.plot(y_test,'g')