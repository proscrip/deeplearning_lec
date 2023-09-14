import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
column_names = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class']
data_set = pd.read_csv('C:/Users/alsgur/Downloads/iris/iris.data', names=column_names)
df_target = pd.DataFrame(data=data_set, columns=['Petal_width'])
print(df_target)
df_data = data_set.drop(columns=['Class', 'Petal_width'], axis=1)
print(data_set.isna().sum())
iris_data =  pd.concat([df_data, df_target], axis=1)
print(iris_data)

x = pd.DataFrame(data=data_set, columns=['Sepal_length', 'Sepal_width', 'Petal_length'])
y = label.fit_transform(iris_data['Petal_width'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=423)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeRegressor

clf_dt = DecisionTreeRegressor()
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print(clf_dt.score(X_train, y_train))

mse = np.sqrt(mean_squared_error(pred_dt, y_test))
print('DecisionTree 평균제곱근오차', mse)

from sklearn.ensemble import RandomForestRegressor

rf_clf = RandomForestRegressor()
rf_clf.fit(X_train, y_train)

pred_rf = rf_clf.predict(X_test)

print(rf_clf.score(X_train, y_train))
mse = np.sqrt(mean_squared_error(pred_rf, y_test))
print('RandomForest 평균제곱근오차', mse)


from sklearn.linear_model import LinearRegression

clf_lr = LinearRegression()
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test)

print(clf_lr.score(X_train, y_train))

mse = np.sqrt(mean_squared_error(pred_lr, y_test))
print('LinearRegression 평균제곱근오차', mse)

from sklearn.svm import SVR

clf_svm = SVR()
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print(clf_svm.score(X_train, y_train))

mse = np.sqrt(mean_squared_error(pred_svm, y_test))
print('SVR 평균제곱근오차', mse)