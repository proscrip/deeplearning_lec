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
data_set = pd.read_excel('C:/Users/alsgu/Downloads/Raisin_Dataset/Raisin_Dataset.xlsx')
df_target = pd.DataFrame(data=data_set, columns=['Class'])
df_target = pd.get_dummies(df_target['Class'])
df_data = data_set.drop(columns=["Class"])
data=pd.concat([df_data, df_target], axis=1)


X = pd.get_dummies(pd.DataFrame(data_set,columns=['Extent']))
Y = df_target
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam



model = Sequential()

model.add(Dense(1,input_shape=(1,),activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(Adam(lr=0.02),'binary_crossentropy',metrics=['accuracy'])

model.summary()

model_history=model.fit(x=X_train, y=y_train, epochs=100, batch_size=32,validation_data= (X_test,y_test))
y_pred = model.predict(X_test)
y_pred = 0.5 < y_pred
print(y_pred)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
model.evaluate(X_test, y_test)
#F1_Score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#LSTM
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(900, 64),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])