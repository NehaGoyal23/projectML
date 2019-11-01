
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
from sklearn.naive_bayes import BernoulliNB


data=pd.read_csv('Gear6SSNE30.csv',header=None)
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
#model = rbf.RBFN(1)
model1.fit(X_train,y_train)

model2 = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

model2.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV


Accuracy = cross_val_score(model, X_train, y_train, cv=10 , scoring="accuracy")

f1_score=cross_val_score(model, X_train, y_train, cv=10, scoring='f1_macro')



Accuracy1 = cross_val_score(model1, X_train, y_train, cv=10 , scoring="accuracy")

f1_score1=cross_val_score(model1, X_train, y_train, cv=10, scoring='f1_macro')


Accuracy2 = cross_val_score(model2, X_train, y_train, cv=10 , scoring="accuracy")

f1_score2=cross_val_score(model2, X_train, y_train, cv=10, scoring='f1_macro')


print('decision tree')

print (Accuracy)
print (f1_score)

print('SVC')

print (Accuracy1)
print (f1_score1)

print('Naive bayes')

print (Accuracy2)
print (f1_score2)

