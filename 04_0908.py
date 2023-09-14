import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
column_names = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class']
data_set = pd.read_csv('C:/Users/alsgur/Downloads/car+evaluation/car.data', names=column_names)
for cov in column_names:
    data_set[cov] = label.fit_transform(data_set[cov])
data_set['Doors'].loc['Doors'] = data_set['Doors']+2
data_set['Persons'].loc['Persons'] = (data_set['Persons']+1)*2
df_target = pd.DataFrame(data=data_set, columns=['Class'])
print(df_target)
df_data = data_set.drop(columns=['Class'], axis=1)
print(data_set.isna().sum())
car_data =  pd.concat([df_data, df_target], axis=1)
print(car_data)
colormap = plt.cm.PuBu
plt.figure(figsize=(8, 8))
plt.title("Analyze Correlation of Features", y = 1.0, size = 16)
sns.heatmap(car_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True, annot_kws={"size" : 16})
plt.show()

x = pd.DataFrame(data=data_set, columns=['Buying', 'Maint', 'Lug_boot'])
y = car_data['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

clf_svm = SVC(random_state=0)
clf_svm.fit(x_train, y_train)
pred_svm = clf_svm.predict(x_test)
print("\n--- SVM Classifier ---")
print(accuracy_score(y_test, pred_svm))
print(confusion_matrix(y_test, pred_svm))

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(x_train, y_train)

pred_lr = clf_lr.predict(x_test)

print ("\n--- Logistic Regression Classifier ---")
print (accuracy_score(y_test, pred_lr))
print (confusion_matrix(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(x_train, y_train)

pred_dt = clf_dt.predict(x_test)

print ("\n--- Decision Tree Classifier ---")
print (accuracy_score(y_test, pred_dt))
print (confusion_matrix(y_test, pred_dt))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print ("\n--- Random Forest ---")
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(x_train, y_train)
pred = rf_clf.predict(x_test)
print(accuracy_score(y_test,pred))
print (confusion_matrix(y_test, pred))