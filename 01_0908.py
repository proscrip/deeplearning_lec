import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  LabelEncoder
label = LabelEncoder()
data_set = pd.read_excel('/Users/alsgur/Downloads/Pumpkin_Seeds_Dataset/Pumpkin_Seeds_Dataset.xlsx')
print(data_set.keys())
df_target = pd.DataFrame(data=data_set, columns=['Class'])
df_target['Class'] = label.fit_transform(df_target['Class'])
print(df_target)
df_data = data_set.drop(columns=['Class'], axis=1)
print(data_set.isna().sum())
pumpkin_data =  pd.concat([df_data, df_target], axis=1)
print(pumpkin_data)
#sns.pairplot(pumpkin_data, vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'], hue ='Class')
#plt.show() Eccentricity, Aspect_Ration'
#colormap = plt.cm.PuBu
#plt.figure(figsize=(8, 8))
#plt.title("Analyze Correlation of Features", y = 1.0, size = 16)
#sns.heatmap(pumpkin_data.astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "black", annot = True, annot_kws = {"size" : 16})
#plt.show()
x = pd.DataFrame(data=data_set, columns=['Eccentricity', 'Aspect_Ration'])
y = pumpkin_data['Class']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

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