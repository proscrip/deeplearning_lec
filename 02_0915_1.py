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
column_names = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class']
data_set = pd.read_csv('C:/Users/alsgu/Downloads/car+evaluation/car.data', names=column_names)
df_target = pd.DataFrame(data=data_set, columns=['Class'])
df_data = data_set.drop(columns=["Class"])

x = pd.get_dummies(df_data.drop(columns=['Doors', 'Persons', 'Safety']))
X = x.iloc[:, :].values
y = data_set.iloc[:,6].values
Y = pd.get_dummies(y)
print(x)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam


model = Sequential()

model.add(Dense(11,input_shape=(11,),activation='relu'))
model.add(Dense(22,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(4,activation='sigmoid'))

model.compile(Adam(lr=0.02),'binary_crossentropy',metrics=['accuracy'])

model.summary()

model_history=model.fit(x=X_train, y=y_train, epochs=100, batch_size=32,validation_data= (X_test,y_test))
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
