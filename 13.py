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

X_train, X_test = train_test_split(df, test_size=0.5, random_state=42)



print(X_train.shape, X_test.shape)

# Normal = 0 Fall = 1

normal = X_train[X_train['activity'] == 0]

y_train = normal['activity']

X_train_normal_train = normal.drop(['activity'], axis=1)

y_test = X_test['activity']

X_test = X_test.drop(['activity'], axis=1)

X_train_ft = X_train_normal_train.values

X_test = X_test.values


generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="tanh")
])
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 모델 컴파일
generator.compile(loss="binary_crossentropy", optimizer="adam")
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
batch_size = 128
epochs = 10

for epoch in range(epochs):
    # 생성기 학습
    generator.trainable = True
    discriminator.trainable = False

    x_fake = generator.predict(np.random.normal(size=(batch_size, 100)))

    loss_g = discriminator.train_on_batch(x_fake, np.ones((batch_size, 1)))

    y_train_normal = y_train[y_train == 0]
    y_train_abnormal = y_train[y_train != 0]
    x_train_normal = X_train[X_train == 0]
    x_train_abnormal = X_train[X_train != 0]
    # 판별기 학습
    generator.trainable = False
    discriminator.trainable = True

    x_real_normal = x_train_normal[np.random.randint(0, x_train_normal.shape[0], size=batch_size)]
    x_real_abnormal = x_train_abnormal[np.random.randint(0, x_train_abnormal.shape[0], size=batch_size)]
    x_fake = generator.predict(np.random.normal(size=(batch_size, 100)))

    loss_d_real_normal = discriminator.train_on_batch(x_real_normal, np.ones((batch_size, 1)))
    loss_d_real_abnormal = discriminator.train_on_batch(x_real_abnormal, np.ones((batch_size, 1)))
    loss_d_fake = discriminator.train_on_batch(x_fake, np.zeros((batch_size, 1)))

    print("Epoch:", epoch, "Generator Loss:", loss_g, "Discriminator Loss Real Normal:", loss_d_real_normal, "Discriminator Loss Real Abnormal:", loss_d_real_abnormal, "Discriminator Loss Fake:", loss_d_fake)

# 이상치 탐색
x_gen = generator.predict(np.random.normal(size=(10, 100)))

# 이미지 출력
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    if y_train[i] == 0:
        plt.title("Normal")
    else:
        plt.title("Abnormal")
    plt.imshow(x_gen[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()