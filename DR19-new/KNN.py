# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 03:02:19 2019

@author: ecs
"""
import pandas as pd  
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
data = pd.read_excel('Datasets/Gear19.xlsx')
data = data.dropna(axis = 1, how ='all') 

#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.90)],axis=1)
print(data.info)
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)

#splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
from sklearn.model_selection import cross_val_score
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
#y_pred1 = optimal_classifier_knn.predict(X_test)
#cm1 = confusion_matrix(y_test, y_pred1)
#print(cn1)
#print('Accuracy using knn ' + str(accuracy_score(y_test, y_pred1)))
Accuracy_knn = cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_knn=cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10,scoring='f1_macro')
#print (f1_score)

print('k- Nearest  Neighbor -----------')
print('optimal parameters:',grid_search1.best_params_)
best_result1 = grid_search1.best_score_
print('Best result : ', best_result1)

j=1
for i in Accuracy_knn:
  print ("Fold-"+ str(j)+" Accuracy - "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy_knn)))

j=1
for i in f1_score_knn:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score_knn)))