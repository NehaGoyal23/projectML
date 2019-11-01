# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:13:22 2019

@author: ecs
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV


data= pd.read_excel('Datasets/Gear18.xlsx')
data1 = data.dropna(axis = 1, how ='all') 

new_data=data1.drop(columns=data1.columns[((data1==0).mean()>0.90)],axis=1)
print(new_data)
#corr_matrix = new_data.corr().abs()
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#new_data.drop(new_data[to_drop], axis=1)
X_dataset= new_data.iloc[:,:-1].values
y_data=new_data.iloc[:,-1].values

#X_dataset = new_data.drop('Class', axis=1)  
#y_data = data['Class']
#X_dataset = new_data.iloc[:, :-1].values  
#y_data = data.iloc[:,130].values
#X_data = sc.fit_transform(X_dataset)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_data, test_size=0.2, random_state=0)

X_train=sc.fit_transform(X_train)
X_test= sc.transform(X_test)


###


#X = sc.fit_transform(X)
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_data, test_size=0.2, random_state=0)



X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#param_grid = {'C': 10, 'gamma' : gammas}
#grid_search = GridSearchCV(estimator = RBFClassifier, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)

#grid_search.fit(X_train, y_train)
#print(grid_search.best_params_)
#best_result = grid_search.best_score_
#print(best_result)
model = SVC(kernel= 'poly',gamma = .22,C = 10)
#model = rbf.RBFN(1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
#cm = confusion_matrix(y_test, y_pred.round())
#print(cm)
#print('Accuracy=' + str(accuracy_score(y_test, y_pred.round(),normalize="False")))
import sklearn

from sklearn.model_selection import cross_val_score

#print(sorted(sklearn.metrics.SCORERS.keys()))

Accuracy = cross_val_score(model, X_train, y_train, cv=10 , scoring="accuracy")
#print (Accuracy)
f1_score=cross_val_score(model, X_train, y_train, cv=10, scoring='f1_macro')
#print (f1_score)

j=1
for i in Accuracy:
  print ("Fold-"+ str(j)+" Accuracy- "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy)))

j=1
for i in f1_score:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score)))