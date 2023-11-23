import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow import stack
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
data = pd.read_csv("C:/Users/alsgur/Downloads/pamap2.csv", index_col=0)
label = LabelEncoder()
data['activityID'] = label.fit_transform(data['activityID'])
act = data[['activityID']]
fhand = data[['handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3']]
fchest = data[['chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestAcc6_1',  'chestAcc6_2',  'chestAcc6_3', 'chestGyro1', 'chestGyro2', 'chestGyro3', 'chestMagne1', 'chestMagne2', 'chestMagne3']]
trainX, trainy, testX, testy = fhand, act, fchest, act

from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam
verbose,epochs,batch_size=1,10,32
label = LabelEncoder()
TIME_STEPS = 67
STEP = 20
def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

X_train, trainy=create_dataset(fhand, act, TIME_STEPS, STEP)
X_test, testy=create_dataset(fchest, act, TIME_STEPS, STEP)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown = "ignore", sparse = False)
enc = enc.fit(trainy)

y_train = enc.transform(trainy)
y_test = enc.transform(testy)

print("X_train.shape: ", trainX.shape)
print("X_test.shape ", testX.shape)
print("y_train.shape ", y_train.shape)
print("y_test.shape ", y_test.shape)
n_timesteps,n_features,n_outputs=X_train.shape[1],X_train.shape[2],y_train.shape[1]
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, LSTM, Dropout, Dense
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

inputs=keras.Input(shape=(n_timesteps,n_features))

conv_1=tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=2,activation='tanh')(inputs)
maxpool_1=tf.keras.layers.MaxPooling1D(pool_size=2,strides=2)(conv_1)

conv_2=tf.keras.layers.Conv1D(filters=96,kernel_size=3,strides=1,activation='relu')(maxpool_1)
avg_pooling=tf.keras.layers.GlobalAveragePooling1D()(conv_2)
batch_norm=tf.keras.layers.BatchNormalization()(avg_pooling)

conv_2=tf.keras.layers.Conv1D(filters=96,kernel_size=3,strides=1,activation='tanh')(maxpool_1)
avg_pooling=tf.keras.layers.GlobalAveragePooling1D()(conv_2)
batch_norm=tf.keras.layers.BatchNormalization()(avg_pooling)

conv_2=tf.keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,activation='relu')(maxpool_1)
avg_pooling=tf.keras.layers.GlobalAveragePooling1D()(conv_2)
batch_norm=tf.keras.layers.BatchNormalization()(avg_pooling)

output=tf.keras.layers.Dense(n_outputs,activation='softmax')(batch_norm)
model=tf.keras.Model(inputs=inputs,outputs=output)
model.summary()

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1, shuffle = True)
print(model.summary())

import matplotlib.pyplot as plt
import seaborn as sns

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.show()


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
base_loss, base_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

score = base_accuracy * 100
print('Accuracy >{:f}'.format(score))
print('Base Loss >{:.2f}'.format(base_loss))