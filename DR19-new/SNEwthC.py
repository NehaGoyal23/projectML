# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:11:48 2019

@author: ecs
"""

# -*- coding: utf-8 -*-

import pandas as pd  
import numpy as np  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
data=pd.read_csv('14GearSSNE10.csv',header=None)
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values
#Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)
print(data)
#splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model1 = SVC(kernel= 'poly',gamma = .22,C = 100)
model1.fit(X_train,y_train)
classifier_knn=KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [1],
    "metric": ["euclidean", "cityblock"],"algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']
}

#tuning parameter of knn
grid_search1 = GridSearchCV(estimator = classifier_knn, param_grid=param_grid_knn, cv = 10, n_jobs = -1, verbose = 2)
grid_search1.fit(X_train,y_train)

optimal_classifier_knn=KNeighborsClassifier(**grid_search1.best_params_)
optimal_classifier_knn.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
Accuracy = cross_val_score(model, X_train, y_train, cv=10 , scoring="accuracy")
f1_score=cross_val_score(model, X_train, y_train, cv=10, scoring='f1_macro')
Accuracy1 = cross_val_score(model1, X_train, y_train, cv=10 , scoring="accuracy")
f1_score1=cross_val_score(model1, X_train, y_train, cv=10, scoring='f1_macro')
Accuracy2 = cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10 , scoring="accuracy")
f1_score2=cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10, scoring='f1_macro')
print('decision tree')
print ('Accuracy',Accuracy)
print ('F-score' ,f1_score)
print('SVM')
print ('Accuracy',Accuracy1)
print ('F-score',f1_score1)
print('KNN')
print ('Accuracy',Accuracy2)
print ('F-score',f1_score2)